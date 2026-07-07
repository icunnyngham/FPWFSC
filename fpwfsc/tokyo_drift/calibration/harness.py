"""Calibration recovery harness (sim only).

Runs the full v1 calibration against a bench sim WITHOUT access to the
injected truth, then compares the fitted profile against
``bench.truth`` and reports per-parameter recovery error. This is the
regression harness for the calibration code itself — it must pass
before any calibration change ships, and it is what the GUI's
"test calibration" action runs.
"""
import numpy as np

from ..dm import TranslationDM
from ..mode_registry import load_ts2_config, mode_n_modes
from ..sim import BenchSim, IdealSim
from ..sim.bench_sim import DM_NOMINAL_SCALE
from ..preprocess import PreprocessImage
from .manual import (
    fit_center,
    fit_dm_scale,
    fit_flips,
    fit_rotation,
    probe_coefficients,
)


def _angle_error_deg(fitted, expected):
    """Minimal signed distance between two angles, degrees."""
    return float((fitted - expected + 180.0) % 360.0 - 180.0)


def acquire_probe(mode_name, *, preset="easy", seed=None, bench=None,
                  ideal=None, average=16):
    """Build the sims (or reuse the given ones), poke the calibration
    probe, and acquire the frames every calibration path starts from.

    Returns a context dict: ``bench, ideal, raw, reference, probe,
    corrector, n_modes, crop_res``. Also what the GUI's View action
    displays before any fitting happens.
    """
    if bench is None:
        bench = BenchSim.from_mode(mode_name, preset=preset, seed=seed)
    if ideal is None:
        ideal = IdealSim.from_mode(mode_name)
    n_modes = mode_n_modes(mode_name)
    corrector = load_ts2_config(mode_name)["corrector_chain"][0]

    probe = probe_coefficients(n_modes)
    translator = TranslationDM(n_modes=n_modes,
                               dm_actuate_scale=DM_NOMINAL_SCALE)
    bench.set_dm_data(translator.command_microns(probe))
    raw = bench.take_image(average=average)
    bench.set_dm_data(translator.command_microns(np.zeros(n_modes)))
    reference = np.asarray(ideal.psf({corrector: probe}))

    return {
        "bench": bench,
        "ideal": ideal,
        "raw": raw,
        "reference": reference,
        "probe": probe,
        "corrector": corrector,
        "n_modes": n_modes,
        "crop_res": reference.shape[0],
    }


def calibrate_bench_sim(mode_name, preset="easy", seed=None, *,
                        average=16, coarse_step=2.0, bench=None,
                        ideal=None, stage_callback=None, around=None,
                        rot_halfwidth=5.0, rot_step=0.1,
                        scale_halfwidth=0.25):
    """Fit a calibration profile against a bench sim; report recovery.

    Returns ``(profile, report)``. The fitters see only what real
    hardware would provide (frames + the DM command channel);
    ``bench.truth`` is touched exclusively for the report.

    ``stage_callback``, if given, is called after each stage with a dict
    ``{stage, params, preview, reference, curve, curve_title}`` so a GUI
    can stream the fitting progress (stages: probe, rotation, center,
    flips, scale).

    ``around`` switches to fine-tune mode: a restricted, finer search
    around an existing profile dict — rotation swept only
    ``±rot_halfwidth`` deg around the current value at ``rot_step``
    resolution, DM scale over ``±scale_halfwidth`` (fractional) on a
    fine grid, and flips kept from the profile (not re-fit). This is
    the fast session-start recalibration (the bench workflow: a narrow
    sweep around the previously saved rotation).
    """
    def _emit(stage, params, preview, curve=None, curve_title=None):
        if stage_callback is not None:
            stage_callback({"stage": stage, "params": dict(params),
                            "preview": preview, "reference": ref,
                            "curve": curve, "curve_title": curve_title})

    # All fitting runs on a known asymmetric probe poke (a null PSF is
    # centro-symmetric — rotation/flips are unidentifiable from it in a
    # clean simulation; see probe_coefficients).
    ctx = acquire_probe(mode_name, preset=preset, seed=seed, bench=bench,
                        ideal=ideal, average=average)
    bench, ideal = ctx["bench"], ctx["ideal"]
    raw, ref = ctx["raw"], ctx["reference"]
    probe, corrector = ctx["probe"], ctx["corrector"]
    crop_res = ctx["crop_res"]
    ideal_psf_fn = lambda coeffs: ideal.psf({corrector: coeffs})  # noqa: E731

    params = {}
    _emit("probe", params, raw)

    if around is None:
        rotation = fit_rotation(raw, ref, crop_res, coarse_step=coarse_step)
    else:
        current_rot = float(around.get("image_rot_deg", 0.0))
        rotation = fit_rotation(
            raw, ref, crop_res,
            angle_range=(current_rot - rot_halfwidth,
                         current_rot + rot_halfwidth),
            coarse_step=max(5.0 * rot_step, 0.5), refine_step=rot_step)
    params["image_rot_deg"] = rotation["image_rot_deg"]
    _emit("rotation", params, rotation["preview"],
          curve=(rotation["angles"], rotation["scores"]),
          curve_title="Rotation sweep score")

    center = fit_center(raw, ref, rotation["image_rot_deg"], crop_res)
    params["crop_cx"] = center["crop_cx"]
    params["crop_cy"] = center["crop_cy"]
    _emit("center", params, center["preview"])

    if around is None:
        flips = fit_flips(raw, ref, rotation["image_rot_deg"],
                          center["crop_cx"], center["crop_cy"], crop_res)
    else:
        # Fine-tune keeps the profile's flip choice (a flip is not a
        # small-parameter-space move).
        kept = PreprocessImage(
            crop_res=crop_res, rot_angle=rotation["image_rot_deg"],
            center_x=center["crop_cx"], center_y=center["crop_cy"],
            flip_horizontal=bool(around.get("flip_x", False)),
            flip_vertical=bool(around.get("flip_y", False)),
            verbose=False)
        flips = {"flip_x": bool(around.get("flip_x", False)),
                 "flip_y": bool(around.get("flip_y", False)),
                 "scores": None,
                 "preview": kept.process(raw, normalize=False)}
    params["flip_x"] = flips["flip_x"]
    params["flip_y"] = flips["flip_y"]
    _emit("flips", params, flips["preview"])

    preprocess = PreprocessImage(
        crop_res=crop_res, rot_angle=rotation["image_rot_deg"],
        center_x=center["crop_cx"], center_y=center["crop_cy"],
        flip_horizontal=flips["flip_x"], flip_vertical=flips["flip_y"],
        verbose=False)
    scale_grid = None
    if around is not None:
        current_scale = float(around.get("dm_scale", 1.0)) or 1.0
        scale_grid = np.geomspace(current_scale * (1.0 - scale_halfwidth),
                                  current_scale * (1.0 + scale_halfwidth),
                                  21)
    scale = fit_dm_scale(preprocess.process(raw, normalize=True),
                         ideal_psf_fn, probe, scale_grid=scale_grid)
    params["dm_scale"] = scale["dm_scale"]
    _emit("scale", params, scale["preview"],
          curve=(scale["scales"], scale["scores"]),
          curve_title="DM-scale match score")

    profile = {
        "mode": mode_name,
        "image_rot_deg": rotation["image_rot_deg"],
        "crop_cx": center["crop_cx"],
        "crop_cy": center["crop_cy"],
        "flip_x": flips["flip_x"],
        "flip_y": flips["flip_y"],
        "dm_scale": scale["dm_scale"],
        "dm_rot_deg": 0.0,   # v1: manual field; automated fit is v2
        "shift_x": 0,
        "shift_y": 0,
    }

    # --- recovery report (the ONLY place truth is read) ---------------
    truth = bench.truth
    expected_rot = (-truth["image_rot_deg"]) % 360.0
    report = {
        "truth": dict(truth),
        "image_rot_error_deg": _angle_error_deg(profile["image_rot_deg"],
                                                expected_rot),
        "dm_scale_error_frac": (profile["dm_scale"] - truth["dm_scale"])
                               / truth["dm_scale"],
        # The bench sim never flips the *image* (DM-side flips are a
        # separate axis, v1-unfitted), so fitted image flips should be
        # False whenever the injected DM flips are too.
        "flips_expected_false": not (profile["flip_x"] or profile["flip_y"]),
        "rotation_curve": (rotation["angles"], rotation["scores"]),
        "scale_curve": (scale["scales"], scale["scores"]),
        "stage_previews": {
            "raw": raw,
            "rotated": rotation["preview"],
            "centered": center["preview"],
            "flipped": flips["preview"],
        },
        "reference_psf": np.asarray(ref),
        "probe_coefficients": probe,
        "corrector": corrector,
    }
    return profile, report
