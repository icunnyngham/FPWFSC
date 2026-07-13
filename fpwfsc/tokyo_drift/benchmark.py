#!/usr/bin/env python
"""Closed-loop timing benchmark: where does an iteration's time go?

Runs the REAL closed loop (``loop.run_closed_loop``, wired exactly like
``run()``'s sim branch) with lightweight timing wrappers around each
seam, then prints a per-stage breakdown:

- ``camera: TS2 render``      one optical propagation per exposure
- ``camera: mangle + noise``  rotate/crop/flux-scale + N Poisson draws
- ``frame reduction``         background / flat / bad-pixel handling
- ``preprocess``              rotate -> center -> crop to the model frame
- ``NN inference``            predictor.predict (device-configurable)
- ``command translation``     mode coefficients -> 50x50 microns command
- ``DM write``                shipping the command to the (sim) DM

The loop code itself is untouched — everything is measured by wrapping
the callables the loop already consumes, so the breakdown reflects the
production wiring (min-max normalization and integrator math land in
"loop overhead", both negligible). On GPU devices the inference wall
time is honest: ``predict`` returns numpy, which forces device sync.

Run from the CLI with a normal run config (the shipped sim ini by
default), e.g.::

    python -m fpwfsc.tokyo_drift.benchmark --mode vampires_vvc_f750_35zern_crop
    python -m fpwfsc.tokyo_drift.benchmark --device mps --iters 20
    python -m fpwfsc.tokyo_drift.benchmark my_run.ini --predictor oracle

With no ``[MODE] calibration profile`` configured, a full calibration
is fitted first (and reported as a one-time cost) so the loop runs
with realistic alignment against the misaligned bench.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..common import support_functions as sf

PIPELINE_DIR = Path(__file__).resolve().parent


class StageTimer:
    """Accumulate wall time per named stage."""

    def __init__(self):
        self.stages = {}  # name -> [total_seconds, calls]

    def add(self, name, seconds):
        entry = self.stages.setdefault(name, [0.0, 0])
        entry[0] += seconds
        entry[1] += 1

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(name, time.perf_counter() - t0)
        return timed


class _TimedPreprocess:
    """PreprocessImage stand-in that times ``process`` calls."""

    def __init__(self, inner, timer, name="preprocess"):
        self._inner = inner
        self._process = timer.wrap(name, inner.process)

    def process(self, *args, **kwargs):
        return self._process(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


class _TimedPredictor:
    def __init__(self, inner, timer, name="NN inference"):
        self._inner = inner
        self.predict = timer.wrap(name, inner.predict)


class _TimedTranslator:
    def __init__(self, inner, timer, name="command translation"):
        self.n_modes = inner.n_modes
        self.command_microns = timer.wrap(name, inner.command_microns)


def run_benchmark(config, configspec, *, mode=None, device=None,
                  n_iter=None, predictor_name=None, error_rms=None):
    """Build the sim-mode loop exactly as ``run()`` does, execute it
    with per-stage timing, and return the results dict."""
    from .calibration.harness import calibrate_bench_sim
    from .calibration.profiles import PROFILE_DEFAULTS, load_profile
    from .dm import DMSafetyBounds, TranslationDM
    from .loop import LeakyIntegrator, run_closed_loop
    from .mode_registry import mode_n_modes, mode_zernike_diameter
    from .predictors import CheatingOracle, RandomWalkPredictor
    from .preprocess import PreprocessImage
    from .run import make_frame_reducer
    from .sim import BenchSim, IdealSim
    from .sim.bench_sim import DM_NOMINAL_SCALE

    settings = sf.validate_config(config, configspec)
    mode_name = mode or settings['MODE']['mode name']
    predictor_name = predictor_name or settings['LOOP_SETTINGS']['predictor']
    device = device or settings['MODEL']['device']
    n_iter = n_iter or settings['LOOP_SETTINGS']['N iter']
    average = settings['SNR']['frames to average']
    flux = 10.0 ** settings['SNR']['int phot flux exponent']

    setup = {}
    timer = StageTimer()

    def timed_setup(name, fn):
        t0 = time.perf_counter()
        result = fn()
        setup[name] = time.perf_counter() - t0
        return result

    n_modes = mode_n_modes(mode_name)
    ideal = timed_setup("IdealSim build", lambda: IdealSim.from_mode(mode_name))
    bench = timed_setup("BenchSim build", lambda: BenchSim.from_mode(
        mode_name, preset=settings['ALIGNMENT']['bench sim preset'],
        seed=settings['SIMULATION']['seed'], int_phot_flux=flux))

    profile_path = settings['MODE']['calibration profile']
    if profile_path:
        profile = load_profile(profile_path)
    else:
        fitted, _ = timed_setup("calibration (no profile configured)",
                                lambda: calibrate_bench_sim(
                                    mode_name, bench=bench, ideal=ideal,
                                    average=average))
        # Fill the unfitted-field defaults load_profile would supply.
        profile = {**PROFILE_DEFAULTS, **fitted}

    translator = TranslationDM(
        n_modes=n_modes,
        dm_actuate_scale=DM_NOMINAL_SCALE / profile["dm_scale"],
        dm_rot_deg=profile["dm_rot_deg"] or None,
        flip_horizontal=profile["dm_flip_x"],
        flip_vertical=profile["dm_flip_y"],
        zernike_diameter=mode_zernike_diameter(mode_name))
    crop_res = ideal.reference_psf.shape[0]
    preprocess = PreprocessImage(
        crop_res=crop_res, rot_angle=profile["image_rot_deg"],
        center_x=profile["crop_cx"], center_y=profile["crop_cy"],
        flip_horizontal=profile["flip_x"], flip_vertical=profile["flip_y"],
        verbose=False)

    rng = np.random.default_rng(settings['SIMULATION']['wfe seed'])
    if error_rms is None:
        error_rms = settings['SIMULATION']['initial error rms']
    error_coeffs = rng.normal(0.0, float(error_rms), n_modes)
    bench.set_modal_error(error_coeffs)
    integrator = LeakyIntegrator(n_modes,
                                 gain=settings['LOOP_SETTINGS']['gain'],
                                 leak=settings['LOOP_SETTINGS']['leak factor'])
    safety = DMSafetyBounds(
        max_ptv_um=settings['DM']['max peak to valley (um)'],
        max_stroke_um=settings['DM']['max actuator stroke (um)'])

    initial_move = None
    if predictor_name == 'model':
        from .model_predictor import TorchPredictor
        predictor = timed_setup("model load", lambda: TorchPredictor.from_mode(
            mode_name, device=device))
        # First inference pays lazy allocation / kernel-compile costs;
        # report it separately so steady-state numbers stay honest.
        timed_setup("first inference (warmup)", lambda: predictor.predict(
            np.zeros((2, crop_res, crop_res)), np.zeros(n_modes)))
        initial_move = rng.normal(
            0.0, settings['MODEL']['initial move sigma'], n_modes)
    elif predictor_name == 'oracle':
        predictor = CheatingOracle(
            residual_fn=lambda: error_coeffs + integrator.state, rng=rng)
    elif predictor_name == 'random_walk':
        predictor = RandomWalkPredictor(n_modes, rng=rng)
    else:
        raise ValueError(f"unknown predictor {predictor_name!r}")

    # --- timing wrappers around the production seams -------------------
    bench._render = timer.wrap("camera: TS2 render", bench._render)
    raw_take = timer.wrap("camera total (render+mangle+noise)",
                          bench.take_image)
    reduce_frame = make_frame_reducer(
        {'bkgd': sf.load_fits_or_none(
            settings['CAMERA CALIBRATION']['background file']),
         'masterflat': sf.load_fits_or_none(
            settings['CAMERA CALIBRATION']['masterflat file']),
         'badpix': sf.load_fits_or_none(
            settings['CAMERA CALIBRATION']['badpix file'])},
        settings['CAMERA CALIBRATION']['estimate background from border'])
    timed_reduce = timer.wrap("frame reduction", reduce_frame)

    def take_image(average=1):
        return timed_reduce(raw_take(average=average))

    send_command = timer.wrap("DM write", bench.set_dm_data)

    t_loop = time.perf_counter()
    result = run_closed_loop(
        take_image, send_command,
        _TimedPredictor(predictor, timer),
        _TimedTranslator(translator, timer),
        integrator, _TimedPreprocess(preprocess, timer),
        n_iter, safety=safety, average=average, initial_move=initial_move)
    loop_wall = time.perf_counter() - t_loop

    residuals = [float(np.linalg.norm(error_coeffs + s))
                 for s in result["states"]]
    return {
        "mode": mode_name, "predictor": predictor_name, "device": device,
        "n_iter": result["iterations"], "average": average, "flux": flux,
        "setup": setup, "stages": timer.stages, "loop_wall": loop_wall,
        "residuals": residuals,
    }


def print_report(r):
    print(f"\ntokyo_drift benchmark — mode {r['mode']}, "
          f"predictor {r['predictor']} (device {r['device']}), "
          f"{r['n_iter']} iterations, average={r['average']}, "
          f"flux={r['flux']:.4g} ph/m^2")

    print("\none-time setup:")
    for name, secs in r["setup"].items():
        print(f"  {name:38s} {secs:8.2f} s")

    # 'camera total' contains 'TS2 render'; report render and the
    # remainder (mangle + Poisson/read noise draws) as its parts.
    stages = dict(r["stages"])
    cam_total, cam_calls = stages.pop("camera total (render+mangle+noise)",
                                      [0.0, 0])
    render, render_calls = stages.pop("camera: TS2 render", [0.0, 0])
    rows = [("camera: TS2 render", render, render_calls),
            ("camera: mangle + noise draws", cam_total - render, cam_calls)]
    rows += [(name, secs, calls) for name, (secs, calls) in stages.items()]
    accounted = cam_total + sum(
        secs for name, (secs, calls) in stages.items())
    rows.append(("loop overhead (normalize, integrator, ...)",
                 r["loop_wall"] - accounted, r["n_iter"]))

    print(f"\nper-stage (loop wall {r['loop_wall']:.2f} s, "
          f"{r['loop_wall'] / max(r['n_iter'], 1) * 1e3:.0f} ms/iter):")
    print(f"  {'stage':42s} {'calls':>5s} {'total':>9s} "
          f"{'mean':>9s} {'share':>6s}")
    for name, secs, calls in rows:
        mean_ms = secs / calls * 1e3 if calls else 0.0
        share = secs / r["loop_wall"] * 100 if r["loop_wall"] else 0.0
        print(f"  {name:42s} {calls:5d} {secs:8.2f} s "
              f"{mean_ms:6.1f} ms {share:5.1f}%")

    res = ", ".join(f"{v:.3f}" for v in r["residuals"])
    print(f"\nmodal residual ||err + state||: {res}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-stage timing of the tokyo_drift closed loop "
                    "(sim mode).")
    parser.add_argument("config", nargs="?",
                        default=str(PIPELINE_DIR / "tokyo_drift_config_sim.ini"),
                        help="run config .ini (default: the shipped sim ini)")
    parser.add_argument("--spec",
                        default=str(PIPELINE_DIR / "tokyo_drift_config.spec"))
    parser.add_argument("--mode", help="override [MODE] mode name")
    parser.add_argument("--device", help="override [MODEL] device "
                                         "(cpu / mps / cuda)")
    parser.add_argument("--iters", type=int, help="override [LOOP] N iter")
    parser.add_argument("--predictor",
                        choices=["model", "oracle", "random_walk"],
                        help="override [LOOP] predictor")
    parser.add_argument("--error-rms", type=float,
                        help="override [SIMULATION] initial error rms "
                             "(the ini default 0.15 suits the 10-mode "
                             "model; use ~0.02-0.05 for 35-mode modes)")
    args = parser.parse_args(argv)

    result = run_benchmark(args.config, args.spec, mode=args.mode,
                           device=args.device, n_iter=args.iters,
                           predictor_name=args.predictor,
                           error_rms=args.error_rms)
    print_report(result)
    return result


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
