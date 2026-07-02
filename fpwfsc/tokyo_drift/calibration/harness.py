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


def calibrate_bench_sim(mode_name, preset="easy", seed=None, *,
                        average=16, coarse_step=2.0, bench=None,
                        ideal=None):
    """Fit a calibration profile against a bench sim; report recovery.

    Returns ``(profile, report)``. The fitters see only what real
    hardware would provide (frames + the DM command channel);
    ``bench.truth`` is touched exclusively for the report.
    """
    if bench is None:
        bench = BenchSim.from_mode(mode_name, preset=preset, seed=seed)
    if ideal is None:
        ideal = IdealSim.from_mode(mode_name)
    crop_res = ideal.reference_psf.shape[0]
    n_modes = mode_n_modes(mode_name)
    corrector = load_ts2_config(mode_name)["corrector_chain"][0]
    ideal_psf_fn = lambda coeffs: ideal.psf({corrector: coeffs})  # noqa: E731

    # All fitting runs on a known asymmetric probe poke (a null PSF is
    # centro-symmetric — rotation/flips are unidentifiable from it in a
    # clean simulation; see probe_coefficients).
    probe = probe_coefficients(n_modes)
    translator = TranslationDM(n_modes=n_modes,
                               dm_actuate_scale=DM_NOMINAL_SCALE)
    bench.set_dm_data(translator.command_microns(probe))
    raw = bench.take_image(average=average)
    bench.set_dm_data(translator.command_microns(np.zeros(n_modes)))
    ref = np.asarray(ideal_psf_fn(probe))

    rotation = fit_rotation(raw, ref, crop_res, coarse_step=coarse_step)
    center = fit_center(raw, ref, rotation["image_rot_deg"], crop_res)
    flips = fit_flips(raw, ref, rotation["image_rot_deg"],
                      center["crop_cx"], center["crop_cy"], crop_res)

    preprocess = PreprocessImage(
        crop_res=crop_res, rot_angle=rotation["image_rot_deg"],
        center_x=center["crop_cx"], center_y=center["crop_cy"],
        flip_horizontal=flips["flip_x"], flip_vertical=flips["flip_y"],
        verbose=False)
    scale = fit_dm_scale(preprocess.process(raw, normalize=True),
                         ideal_psf_fn, probe)

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
    }
    return profile, report
