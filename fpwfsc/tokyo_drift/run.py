#!/usr/bin/env python
"""Tokyo Drift — NN-driven focal-plane wavefront sensing & control loop.

The Tokyo Drift method predicts the wavefront error from two consecutive
focal-plane frames plus the DM actuation applied between them (an NN
analogue of Fast & Furious), then corrects with a leaky integrator.

This module is the pipeline entry point, following the same contract as
the other FPWFSC pipelines: ``run()`` reads a validated .ini config and
executes the control loop; the GUI, the command line, and notebooks all
call this same function.
"""
import sys
import threading
from pathlib import Path

import numpy as np

from ..common import support_functions as sf


def run(camera=None, aosystem=None, config=None, configspec=None,
        my_event=None, plotter=None):
    """Run the Tokyo Drift control loop.

    Parameters
    ----------
    camera, aosystem
        ``'Sim'`` string sentinels for simulation mode, or real-hardware
        wrapper instances (see ``gui_helper.load_instruments``).
    config, configspec
        Path (or dict) for the .ini config and its .spec validator.
    my_event
        ``threading.Event`` used by the GUI to signal stop.
    plotter
        Optional live plotter (``tokyo_drift_plotter_qt.LivePlotter``).

    Returns
    -------
    dict with ``settings`` (the validated config) and ``loop`` (the
    closed-loop result dict, or None if stopped before starting).
    """
    if my_event is None:
        my_event = threading.Event()
    settings = sf.validate_config(config, configspec)

    mode_name = settings['MODE']['mode name']
    calibration_profile = settings['MODE']['calibration profile']

    n_iter = settings['LOOP_SETTINGS']['N iter']
    gain = settings['LOOP_SETTINGS']['gain']
    leak_factor = settings['LOOP_SETTINGS']['leak factor']
    strehl_early_stop = settings['LOOP_SETTINGS']['strehl early stop']
    predictor_name = settings['LOOP_SETTINGS']['predictor']

    max_ptv_um = settings['DM']['max peak to valley (um)']
    max_stroke_um = settings['DM']['max actuator stroke (um)']

    preset_name = settings['SIMULATION']['bench sim preset']
    seed = settings['SIMULATION']['seed']
    initial_error_rms = settings['SIMULATION']['initial error rms']

    print(f"tokyo_drift: config OK - mode '{mode_name}', "
          f"{n_iter} iterations requested.")

    if not (camera == 'Sim' and aosystem == 'Sim'):
        raise NotImplementedError(
            "tokyo_drift real-hardware backends are not implemented yet; "
            "run with camera='Sim', aosystem='Sim'")

    if my_event.is_set():
        return {"settings": settings, "loop": None}

    # ------------------------------------------------------------------
    # Simulation-mode backends
    # ------------------------------------------------------------------
    from .dm import DMSafetyBounds, TranslationDM
    from .loop import LeakyIntegrator, peak_flux_ratio, run_closed_loop
    from .mode_registry import mode_n_modes
    from .predictors import CheatingOracle, RandomWalkPredictor
    from .preprocess import PreprocessImage
    from .sim import BenchSim, BenchSimAO, BenchSimCamera, IdealSim
    from .sim.bench_sim import DM_NOMINAL_SCALE

    rng = np.random.default_rng(seed)
    n_modes = mode_n_modes(mode_name)

    ideal = IdealSim.from_mode(mode_name)
    bench = BenchSim.from_mode(mode_name, preset=preset_name, seed=seed)
    Camera = BenchSimCamera(bench)
    AOsystem = BenchSimAO(bench)
    truth = bench.truth
    print(f"tokyo_drift: bench-sim injected truth (sim-only): {truth}")

    # Calibration is a separate, explicit step: the loop only ever
    # consumes a saved profile (fit one with the calibration harness /
    # GUI workbench). No profile -> no loop.
    if calibration_profile is None:
        raise ValueError(
            "no calibration profile selected: fit one (GUI Calibrate "
            "button, or calibration.harness.calibrate_bench_sim) and set "
            "[MODE] 'calibration profile'")
    from .calibration.profiles import assert_sim_safe, load_profile
    profile = load_profile(calibration_profile)
    profile_mode = profile.get("mode")
    if profile_mode is not None and profile_mode != mode_name:
        raise ValueError(
            f"calibration profile is for mode {profile_mode!r}, "
            f"not {mode_name!r}")
    assert_sim_safe(profile)  # shifts are bench-only; refuse in sim

    translator = TranslationDM(
        n_modes=n_modes,
        dm_actuate_scale=DM_NOMINAL_SCALE / profile["dm_scale"],
        dm_rot_deg=profile["dm_rot_deg"] or None,
        flip_horizontal=profile["dm_flip_x"],
        flip_vertical=profile["dm_flip_y"],
    )
    preprocess = PreprocessImage(
        crop_res=ideal.reference_psf.shape[0],
        rot_angle=profile["image_rot_deg"],
        center_x=profile["crop_cx"],
        center_y=profile["crop_cy"],
        flip_horizontal=profile["flip_x"],
        flip_vertical=profile["flip_y"],
    )

    # Reference peak-flux ratio from the pristine bench (before the
    # hidden error is injected) - the denominator of the Strehl proxy.
    reference_frame = preprocess.process(bench.take_image_noiseless(),
                                         normalize=False)
    reference_ratio = peak_flux_ratio(reference_frame)

    # Inject the hidden wavefront error the loop must correct ("the
    # DM's flat isn't flat"), expressed in the modal basis via a
    # nominal, unrotated translator.
    error_coeffs = rng.normal(0.0, initial_error_rms, n_modes)
    error_translator = TranslationDM(n_modes=n_modes,
                                     dm_actuate_scale=DM_NOMINAL_SCALE)
    bench.set_error_command(error_translator.command_microns(error_coeffs))

    integrator = LeakyIntegrator(n_modes, gain=gain, leak=leak_factor)
    if predictor_name == 'oracle':
        predictor = CheatingOracle(
            residual_fn=lambda: error_coeffs + integrator.state, rng=rng)
    elif predictor_name == 'random_walk':
        predictor = RandomWalkPredictor(n_modes, rng=rng)
    else:
        raise ValueError(f"unknown predictor {predictor_name!r}")

    safety = DMSafetyBounds(max_ptv_um=max_ptv_um,
                            max_stroke_um=max_stroke_um)

    result = run_closed_loop(
        Camera.take_image, AOsystem.set_dm_data,
        predictor, translator, integrator, preprocess, n_iter,
        safety=safety,
        strehl_fn=lambda frame: peak_flux_ratio(frame) / reference_ratio,
        strehl_early_stop=strehl_early_stop,
        stop_event=my_event,
        plotter=plotter,
        ideal_psf=ideal.reference_psf,
    )
    print(f"tokyo_drift: loop finished after {result['iterations']} "
          f"iterations.")
    return {"settings": settings, "loop": result, "truth": truth,
            "injected_error_coeffs": error_coeffs}


if __name__ == "__main__":
    camera = "Sim"
    aosystem = "Sim"

    script_dir = Path(__file__).parent
    config_path = script_dir / "tokyo_drift_config_sim.ini"
    spec_path = script_dir / "tokyo_drift_config.spec"

    my_event = threading.Event()
    if "--check-config" in sys.argv:
        my_event.set()  # validate and exit before building backends

    run(camera, aosystem, config=str(config_path),
        configspec=str(spec_path), my_event=my_event, plotter=None)
