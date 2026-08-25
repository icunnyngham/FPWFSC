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
import datetime
import re
import sys
import threading
from pathlib import Path

import numpy as np

from ..common import support_functions as sf


def _filter_number(name):
    """Leading number in a filter name, as a string (None if none).

    Tolerant of naming-convention differences between the mode manifest
    ('F750') and the camera keyword ('750-50'): both reduce to '750'.
    """
    match = re.search(r'(\d+)', str(name))
    return match.group(1) if match else None


def assert_camera_matches_mode(camera_obj, mode_name):
    """Refuse to run a filter-specific NN against the wrong filter.

    tokyo_drift locks wavelength / pixel scale to the trained mode
    rather than reading camera keywords, so the one thing that MUST be
    checked at connect time is that the camera's current filter is the
    one the checkpoint was trained on.
    """
    from .mode_registry import load_manifest
    want = load_manifest(mode_name).get('filter')
    have = getattr(camera_obj, 'filter_name', None)
    if not want:
        return
    if have is None:
        import warnings
        warnings.warn(
            f"camera exposes no filter_name; cannot verify it matches "
            f"mode {mode_name!r} (trained on {want!r})")
        return
    if _filter_number(want) != _filter_number(have):
        raise ValueError(
            f"camera filter {have!r} does not match mode {mode_name!r} "
            f"(trained on {want!r}); change the filter or pick the "
            f"matching mode")
    print(f"tokyo_drift: camera filter {have!r} matches mode filter "
          f"{want!r}")


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
        my_event=None, plotter=None, injector=None):
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
    injector
        Optional DM wrapper (``set_dm_data``) for the WFE-injection
        channel ([SIMULATION] 'injection dm channel', hardware only).
        When None and a channel is configured, one is built as
        ``type(aosystem)(dm_channel=<channel>)`` — only the raw
        ``set_dm_data`` is used, so the wrapper's geometry parameters
        are irrelevant.

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
    n_repeats = settings['SIMULATION']['n repeats']
    injection_channel = settings['SIMULATION']['injection dm channel']

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
    save_camera_frames = settings['IO']['save camera frames']
    hitchhiker_mode = settings['IO']['hitchhiker mode']
    hitchhiker_path = settings['IO']['hitchhiker path']

    print(f"tokyo_drift: config OK - mode '{mode_name}', "
          f"{n_iter} iterations requested.")

    sim_mode = (camera == 'Sim' and aosystem == 'Sim')

    if my_event.is_set():
        return {"settings": settings, "loop": None}

    # ------------------------------------------------------------------
    # Backends (simulated bench, or real-hardware wrapper instances)
    # ------------------------------------------------------------------
    from .dm import DMSafetyBounds, DMSafetyError, TranslationDM
    from .loop import LeakyIntegrator, peak_flux_ratio, run_closed_loop
    from .mode_registry import mode_n_modes, mode_zernike_diameter
    from .predictors import CheatingOracle, RandomWalkPredictor
    from .preprocess import PreprocessImage
    from .sim import IdealSim
    from .sim.bench_sim import DM_NOMINAL_SCALE

    # Two seeds, two owners: `seed` pins the BENCH (misalignment truth +
    # detector noise), so a saved calibration stays valid run to run;
    # `wfe_seed` pins the injected-WFE episode (the error draw + the
    # model's initial diversity move). The default None gives each run a
    # fresh injected error against the same bench.
    rng = np.random.default_rng(wfe_seed)
    n_modes = mode_n_modes(mode_name)

    ideal = IdealSim.from_mode(mode_name)
    if sim_mode:
        from .sim import BenchSim, BenchSimAO, BenchSimCamera
        bench = BenchSim.from_mode(mode_name, preset=preset_name, seed=seed,
                                   int_phot_flux=10.0 ** flux_exponent)
        Camera = BenchSimCamera(bench)
        AOsystem = BenchSimAO(bench)
        truth = bench.truth
        print(f"tokyo_drift: bench-sim injected truth (sim-only): {truth}")
    else:
        # Real hardware: wrapper instances built by the caller (see
        # gui_helper.load_instruments). The camera must match the mode
        # the NN was trained for — the model is filter-specific.
        Camera = camera
        AOsystem = aosystem
        truth = None
        assert_camera_matches_mode(Camera, mode_name)
        # A dark fetched from the camera's shm stream stands in for a
        # background file, unless one was configured explicitly. The
        # reducer stays the single place backgrounds get subtracted.
        camera_dark = getattr(Camera, 'dark', None)
        if camera_dark is not None and bgds['bkgd'] is None:
            bgds['bkgd'] = np.asarray(camera_dark, dtype=float)
            print(f"tokyo_drift: using camera dark "
                  f"({getattr(Camera, 'dark_info', 'no info')})")
        elif camera_dark is None and bgds['bkgd'] is None:
            print("tokyo_drift: WARNING - no camera dark and no "
                  "background file; frames will be reduced without "
                  "background subtraction")

    # Hardware WFE injection: a second DMcomb channel carries the drawn
    # error (the DM sums its channels), so the loop fights a KNOWN
    # injected wavefront instead of only the bench's natural NCPA.
    if injection_channel and sim_mode:
        print("tokyo_drift: 'injection dm channel' is ignored in sim "
              "(the sim injects through the optical model)")
        injector = None
    elif injection_channel and not sim_mode:
        if injection_channel == settings['DM']['dm channel']:
            raise ValueError(
                "injection dm channel must differ from the correction "
                f"'dm channel' ({injection_channel!r}): the loop would "
                "overwrite its own injection")
        if injector is None:
            # Same wrapper class as the correction DM; only the raw
            # set_dm_data path is used, so geometry params don't matter.
            injector = type(AOsystem)(dm_channel=injection_channel)
        print(f"tokyo_drift: hardware WFE injection enabled on "
              f"{injection_channel!r} (rms {initial_error_rms})")
    else:
        injector = None

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
    if sim_mode:
        assert_sim_safe(profile)  # shifts are bench-only; refuse in sim

    translator = TranslationDM(
        n_modes=n_modes,
        dm_actuate_scale=DM_NOMINAL_SCALE / profile["dm_scale"],
        dm_rot_deg=profile["dm_rot_deg"] or None,
        flip_horizontal=profile["dm_flip_x"],
        flip_vertical=profile["dm_flip_y"],
        shift_x=profile.get("shift_x", 0) or 0,
        shift_y=profile.get("shift_y", 0) or 0,
        command_aperture_act=settings['DM']['command aperture actuators'],
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

    # Reference frame: the diffraction-limited PSF, the denominator of
    # both Strehl estimates. In sim it comes from the pristine bench
    # (before the hidden error is injected); on hardware there is no
    # noiseless bench, so the mode's ideal-model PSF stands in — the
    # same reference the NN was trained against.
    if sim_mode:
        reference_frame = preprocess.process(
            reduce(bench.take_image_noiseless()), normalize=False)
    else:
        reference_frame = np.asarray(ideal.reference_psf, dtype=float)
    if strehl_method == 'vandam':
        from ..common import vandamstrehl as vd

        def strehl_fn(frame):
            return float(vd.strehl(frame, reference_frame))
    else:
        reference_ratio = peak_flux_ratio(reference_frame)

        def strehl_fn(frame):
            return peak_flux_ratio(frame) / reference_ratio

    # Predictors whose construction is expensive or stateless are built
    # once and shared across episodes; the oracle is rebuilt per episode
    # (its closure must track that episode's error and integrator).
    if predictor_name == 'oracle':
        if not sim_mode:
            raise ValueError(
                "the 'oracle' predictor needs the sim's injected truth; "
                "on hardware use predictor = 'model'")
        predictor = None
    elif predictor_name == 'random_walk':
        predictor = RandomWalkPredictor(n_modes, rng=rng)
    elif predictor_name == 'model':
        from .model_predictor import TorchPredictor
        predictor = TorchPredictor.from_mode(mode_name, device=model_device)
        print(f"tokyo_drift: loaded model checkpoint "
              f"(run {predictor.run_id}, {predictor.n_modes} modes) on "
              f"device {model_device}")
    else:
        raise ValueError(f"unknown predictor {predictor_name!r}")

    safety = DMSafetyBounds(max_ptv_um=max_ptv_um,
                            max_stroke_um=max_stroke_um)

    # Pre-reduction readout cubes only exist where a camera wrapper
    # exposes them (Vampires stashes each grab as `last_frames`); in sim
    # and hitchhiker mode there is no camera readout to keep.
    if save_camera_frames and (sim_mode or hitchhiker_mode
                               or not hasattr(Camera, 'last_frames')):
        save_camera_frames = False
        print("tokyo_drift: WARNING - 'save camera frames' is on but no "
              "pre-reduction readouts are available (sim / hitchhiker / "
              "camera without last_frames); ignoring")

    # One stamp for the whole invocation: repeated episodes get _rNN
    # suffixes under it, so back-to-back sessions (well under the 1 s
    # stamp resolution once setup is amortized) can never collide.
    session_stamp = datetime.datetime.now().strftime(
        "tokyo_drift_%Y-%m-%dT%H-%M-%S")

    # ------------------------------------------------------------------
    # Episodes: [SIMULATION] 'n repeats' full loop runs against the
    # backends built above. Each episode is logged as its own session.
    # ------------------------------------------------------------------
    episode_results = []
    error_coeffs = None
    try:
        for episode in range(n_repeats):
            if my_event.is_set():
                break
            if n_repeats > 1:
                print(f"tokyo_drift: episode {episode + 1}/{n_repeats}")
            if episode > 0:
                # The loop images before it commands and never resets the
                # DM, so each new episode must start it from zero (through
                # the loop's own command path).
                AOsystem.set_dm_data(
                    translator.command_microns(np.zeros(n_modes)))

            # Inject the hidden wavefront error the loop must correct. In
            # sim it is an EXTERNAL (NCPA-like) aberration in the training
            # modal basis, NOT routed through the DM: cancelling it requires
            # the DM's effective command-to-wavefront gain, which is what
            # the calibrated dm_scale measures (the same physics the real
            # bench absorbed into dm_actuate_scale ~1.4e-6 against a 1e-6
            # nominal). set_modal_error replaces the previous episode's
            # error. On hardware with an injector, the same modal draw is
            # synthesized through the loop's own command translation and
            # written to the injection DMcomb channel — a DM-borne error
            # (through the same influence functions as corrections), the
            # 2024 bench-session methodology. Without an injector the real
            # NCPA plays this role and nothing is injected or recorded.
            episode_abort = None
            injected_command = None
            if sim_mode:
                error_coeffs = rng.normal(0.0, initial_error_rms, n_modes)
                bench.set_modal_error(error_coeffs)
            elif injector is not None:
                error_coeffs = rng.normal(0.0, initial_error_rms, n_modes)
                injection = translator.command_microns(error_coeffs)
                try:
                    safety.check(injection)
                except DMSafetyError as exc:
                    # Configured amplitude drew a command over the DM
                    # limits: never sent; abort this episode like any other
                    # safety trip and let the remaining episodes continue.
                    episode_abort = f"injected WFE command: {exc}"
                    print(f"tokyo_drift: SAFETY ABORT - {episode_abort}")
                else:
                    injector.set_dm_data(injection)
                    injected_command = injection

            integrator = LeakyIntegrator(n_modes, gain=gain, leak=leak_factor)
            # The NN needs a known diversity move before its first
            # prediction (two otherwise-identical frames carry no temporal
            # cue); the dummy predictors ignore actuation, so they run with
            # no initial move.
            initial_move = None
            if predictor_name == 'oracle':
                predictor = CheatingOracle(
                    residual_fn=lambda ec=error_coeffs, it=integrator:
                        ec + it.state,
                    rng=rng)
            elif predictor_name == 'model':
                initial_move = rng.normal(0.0, model_initial_move_sigma,
                                          n_modes)

            logger = None
            iteration_callback = None
            if save_log:
                from .session_log import SessionLogger
                session_name = (f"{session_stamp}_r{episode:02d}"
                                if n_repeats > 1 else session_stamp)
                logger = SessionLogger(log_path, settings=settings,
                                       session_name=session_name)
                # The profile and the subtracted background travel with the
                # log: config.json holds the profile only by path (possibly
                # a GUI tempfile), and the dark's shm buffer gets
                # overwritten.
                logger.save_provenance(profile_path=calibration_profile,
                                       background=bgds['bkgd'])
                logger.save_episode(episode=episode, n_repeats=n_repeats,
                                    injected_error_coeffs=error_coeffs,
                                    initial_move=initial_move,
                                    injected_command=injected_command)
                print(f"tokyo_drift: logging session to {logger.session_dir}")

                def iteration_callback(payload, logger=logger):
                    sent = payload.get("command_sent", True)
                    logger.save_iteration(
                        payload["iteration"],
                        strehl=payload["strehl"],
                        state=payload["state"],
                        prediction=payload["prediction"],
                        dm_command=payload["command"] if sent else None,
                        dm_command_refused=(None if sent
                                            else payload["command"]),
                        safety_error=payload.get("safety_error"),
                        raw=payload["raw"],
                        processed=payload["processed"],
                        camera_frames=(Camera.last_frames
                                       if save_camera_frames else None),
                    )

            if episode_abort is not None:
                # The injection itself was refused: the loop never ran.
                result = {"strehls": np.full(n_iter, np.nan),
                          "states": np.zeros((n_iter, n_modes)),
                          "final_state": np.zeros(n_modes),
                          "iterations": 0,
                          "aborted": "dm_safety",
                          "safety_error": episode_abort}
            else:
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
            result["injected_error_coeffs"] = error_coeffs
            result["initial_move"] = initial_move
            if logger is not None:
                logger.finalize(result)
            episode_results.append(result)

            if result["aborted"]:
                # A refused command was never sent, but the last SENT command
                # was by construction near the limits - never the state to
                # leave parked on the DM. Zero it (a converged run, by
                # contrast, deliberately leaves its solution on the DM for
                # the save-flat workflow).
                AOsystem.set_dm_data(
                    translator.command_microns(np.zeros(n_modes)))
                print(f"tokyo_drift: episode aborted on DM safety after "
                      f"{result['iterations']} iterations; DM zeroed "
                      f"({result['safety_error']})")
                # Every episode aborting from the start is systematic (bad
                # gain / calibration), not unlucky WFE draws.
                if (len(episode_results) == 3
                        and all(r["aborted"] for r in episode_results)):
                    print("tokyo_drift: first 3 episodes all aborted on DM "
                          "safety - stopping the run (check gain and "
                          "calibration)")
                    break
            else:
                print(f"tokyo_drift: loop finished after "
                      f"{result['iterations']} iterations.")

    finally:
        # An injection run must never leave the bench aberrated:
        # whatever ends it (completion, stop, abort, exception),
        # clear the injection channel AND the correction channel -
        # with the injection gone, the converged correction would
        # itself aberrate the bench (and it is episode-specific,
        # not a reusable flat). NCPA-only runs keep the current
        # behavior: the converged state stays on the DM.
        if injector is not None:
            zero = translator.command_microns(np.zeros(n_modes))
            injector.set_dm_data(zero)
            AOsystem.set_dm_data(zero)
            print("tokyo_drift: injection and correction channels "
                  "zeroed")

    n_aborted = sum(1 for r in episode_results if r["aborted"])
    if n_aborted and len(episode_results) > 1:
        print(f"tokyo_drift: {n_aborted}/{len(episode_results)} episodes "
              f"aborted on DM safety")

    return {"settings": settings,
            "loop": episode_results[-1] if episode_results else None,
            "repeats": episode_results,
            "truth": truth,
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
