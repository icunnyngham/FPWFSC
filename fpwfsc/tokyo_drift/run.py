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


def make_frame_reducer(bgds, estimate_background_from_border):
    """Build the raw-frame reduction callable (background / flat / bad
    pixels).

    ``bgds`` holds the loaded calibration arrays (any of which may be
    None). When no background frame is configured,
    ``estimate_background_from_border`` chooses between
    ``equalize_image``'s border-median fallback (True) and no background
    subtraction at all (False, the default): the NN was trained on
    unsubtracted min-max-normalized frames, and on faint coronagraphic
    frames the border estimate subtracts real halo flux and injects
    correlated stripe noise.
    """
    def reduce(frame):
        for name, arr in bgds.items():
            if arr is not None and arr.shape != frame.shape:
                raise ValueError(
                    f"calibration file {name!r} has shape {arr.shape} but "
                    f"the camera frame is {frame.shape}")
        bkgd = bgds['bkgd']
        if bkgd is None and not estimate_background_from_border:
            bkgd = 0.0
        return sf.equalize_image(frame, bkgd=bkgd,
                                 masterflat=bgds['masterflat'],
                                 badpix=bgds['badpix'])
    return reduce


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
    strehl_method = settings['LOOP_SETTINGS']['strehl method']

    model_initial_move_sigma = settings['MODEL']['initial move sigma']
    model_device = settings['MODEL']['device']

    max_ptv_um = settings['DM']['max peak to valley (um)']
    max_stroke_um = settings['DM']['max actuator stroke (um)']

    preset_name = settings['ALIGNMENT']['bench sim preset']
    seed = settings['SIMULATION']['seed']
    wfe_seed = settings['SIMULATION']['wfe seed']
    initial_error_rms = settings['SIMULATION']['initial error rms']

    flux_exponent = settings['SNR']['int phot flux exponent']
    frames_to_average = settings['SNR']['frames to average']

    bgds = {
        'bkgd': sf.load_fits_or_none(
            settings['CAMERA CALIBRATION']['background file']),
        'masterflat': sf.load_fits_or_none(
            settings['CAMERA CALIBRATION']['masterflat file']),
        'badpix': sf.load_fits_or_none(
            settings['CAMERA CALIBRATION']['badpix file']),
    }
    estimate_border = settings['CAMERA CALIBRATION'][
        'estimate background from border']

    save_log = settings['IO']['save_log']
    log_path = settings['IO']['log_path']
    hitchhiker_mode = settings['IO']['hitchhiker mode']
    hitchhiker_path = settings['IO']['hitchhiker path']

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
    from .mode_registry import mode_n_modes, mode_zernike_diameter
    from .predictors import CheatingOracle, RandomWalkPredictor
    from .preprocess import PreprocessImage
    from .sim import BenchSim, BenchSimAO, BenchSimCamera, IdealSim
    from .sim.bench_sim import DM_NOMINAL_SCALE

    # Two seeds, two owners: `seed` pins the BENCH (misalignment truth +
    # detector noise), so a saved calibration stays valid run to run;
    # `wfe_seed` pins the injected-WFE episode (the error draw + the
    # model's initial diversity move). The default None gives each run a
    # fresh injected error against the same bench.
    rng = np.random.default_rng(wfe_seed)
    n_modes = mode_n_modes(mode_name)

    ideal = IdealSim.from_mode(mode_name)
    bench = BenchSim.from_mode(mode_name, preset=preset_name, seed=seed,
                               int_phot_flux=10.0 ** flux_exponent)
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
        zernike_diameter=mode_zernike_diameter(mode_name),
    )
    preprocess = PreprocessImage(
        crop_res=ideal.reference_psf.shape[0],
        rot_angle=profile["image_rot_deg"],
        center_x=profile["crop_cx"],
        center_y=profile["crop_cy"],
        flip_horizontal=profile["flip_x"],
        flip_vertical=profile["flip_y"],
    )

    # Frame reduction (background / flat / bad pixels), matching the
    # other pipelines' reduce-before-use convention (see
    # make_frame_reducer for the border-estimation policy).
    reduce = make_frame_reducer(bgds, estimate_border)

    if hitchhiker_mode:
        from ..common import fake_hardware as fhw
        hitch = fhw.Hitchhiker(imagedir=hitchhiker_path)
        print(f"tokyo_drift: hitchhiker mode - reading frames from "
              f"{hitchhiker_path}")

        def take_image(average=1):
            return reduce(hitch.wait_for_next_image())
    else:
        def take_image(average=1):
            return reduce(Camera.take_image(average=average))

    # Reference frame from the pristine bench (before the hidden error
    # is injected): the diffraction-limited PSF of THIS optical system,
    # the denominator of both Strehl estimates.
    reference_frame = preprocess.process(reduce(bench.take_image_noiseless()),
                                         normalize=False)
    if strehl_method == 'vandam':
        from ..common import vandamstrehl as vd

        def strehl_fn(frame):
            return float(vd.strehl(frame, reference_frame))
    else:
        reference_ratio = peak_flux_ratio(reference_frame)

        def strehl_fn(frame):
            return peak_flux_ratio(frame) / reference_ratio

    # Inject the hidden wavefront error the loop must correct — an
    # EXTERNAL (NCPA-like) aberration in the training modal basis, NOT
    # routed through the DM: cancelling it requires the DM's effective
    # command-to-wavefront gain, which is what the calibrated dm_scale
    # measures (the same physics the real bench absorbed into
    # dm_actuate_scale ~1.4e-6 against a 1e-6 nominal).
    error_coeffs = rng.normal(0.0, initial_error_rms, n_modes)
    bench.set_modal_error(error_coeffs)

    integrator = LeakyIntegrator(n_modes, gain=gain, leak=leak_factor)
    # The NN needs a known diversity move before its first prediction (two
    # otherwise-identical frames carry no temporal cue); the dummy
    # predictors ignore actuation, so they run with no initial move.
    initial_move = None
    if predictor_name == 'oracle':
        predictor = CheatingOracle(
            residual_fn=lambda: error_coeffs + integrator.state, rng=rng)
    elif predictor_name == 'random_walk':
        predictor = RandomWalkPredictor(n_modes, rng=rng)
    elif predictor_name == 'model':
        from .model_predictor import TorchPredictor
        predictor = TorchPredictor.from_mode(mode_name, device=model_device)
        print(f"tokyo_drift: loaded model checkpoint "
              f"(run {predictor.run_id}, {predictor.n_modes} modes) on "
              f"device {model_device}")
        initial_move = rng.normal(0.0, model_initial_move_sigma, n_modes)
    else:
        raise ValueError(f"unknown predictor {predictor_name!r}")

    safety = DMSafetyBounds(max_ptv_um=max_ptv_um,
                            max_stroke_um=max_stroke_um)

    logger = None
    iteration_callback = None
    if save_log:
        from .session_log import SessionLogger
        logger = SessionLogger(log_path, settings=settings)
        print(f"tokyo_drift: logging session to {logger.session_dir}")

        def iteration_callback(payload):
            logger.save_iteration(
                payload["iteration"],
                strehl=payload["strehl"],
                state=payload["state"],
                prediction=payload["prediction"],
                dm_command=payload["command"],
                raw=payload["raw"],
                processed=payload["processed"],
            )

    result = run_closed_loop(
        take_image, AOsystem.set_dm_data,
        predictor, translator, integrator, preprocess, n_iter,
        safety=safety,
        average=frames_to_average,
        initial_move=initial_move,
        strehl_fn=strehl_fn,
        strehl_early_stop=strehl_early_stop,
        stop_event=my_event,
        plotter=plotter,
        ideal_psf=ideal.reference_psf,
        iteration_callback=iteration_callback,
    )
    if logger is not None:
        logger.finalize(result)
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
