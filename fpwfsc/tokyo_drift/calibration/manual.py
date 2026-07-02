"""v1 calibration engine: staged fitters for instrument alignment.

Automates the workflow that was proven manually on the SCExAO bench
(2023-2024): find the camera-vs-model rotation by sweeping applied
rotation angles and maximizing the convolution score against a
reference (simulated) PSF, refine the PSF center by
convolution-with-reference, resolve image flips by scoring the four
combinations, and estimate the DM actuation scale by matching a single
mode poke against ideal-sim renders over a scale grid.

Each fitter returns a dict containing the fitted parameter(s), the
diagnostic curve where applicable, and a ``preview`` frame — the
processed image at that stage — so a GUI can stream the stages to its
alignment panels (raw -> rotated -> centered -> flipped).

Upstream note: fnf ships ``rotation_flip_calibration.py`` /
``run_test_get_rot.py``, but both are manual eyeball tools (poke a
mode, show theory-vs-bench, human infers orientation) — there is no
automated fitting to reuse, so this module implements the bench-proven
conv-sweep directly.
"""
import numpy as np
from scipy.signal import fftconvolve

from ..preprocess import PreprocessImage


def alignment_score(crop, ref_psf):
    """Peak of the cross-CORRELATION between a processed frame and the
    reference PSF (both sum-normalized), FFT-computed.

    The bench used ``convolve2d`` here, which flips the template — for
    a centro-symmetric reference the two are identical, but for the
    asymmetric calibration probe convolution locks onto the
    180°-rotated orientation. Correlation is the correct
    template-matching form.
    """
    a = np.asarray(crop, dtype=float)
    b = np.asarray(ref_psf, dtype=float)
    if a.sum() > 0:
        a = a / a.sum()
    if b.sum() > 0:
        b = b / b.sum()
    return float(fftconvolve(a, b[::-1, ::-1], mode="same").max())


def fit_rotation(raw_frame, ref_psf, crop_res, *, angle_range=(0.0, 360.0),
                 coarse_step=2.0, refine_step=0.2):
    """Fit the rotation to APPLY to raw frames (bench convention).

    Coarse global sweep (the full-circle ambiguity cannot be descended
    through) followed by a local refinement around the coarse argmax.
    """
    def score_at(angle):
        proc = PreprocessImage(crop_res=crop_res, rot_angle=angle,
                               verbose=False)
        return alignment_score(proc.process(raw_frame, normalize=False),
                               ref_psf)

    coarse = np.arange(angle_range[0], angle_range[1], coarse_step)
    coarse_scores = np.array([score_at(a) for a in coarse])
    best_coarse = coarse[int(np.argmax(coarse_scores))]

    fine = np.arange(best_coarse - coarse_step,
                     best_coarse + coarse_step + refine_step / 2, refine_step)
    fine_scores = np.array([score_at(a) for a in fine])
    best = float(fine[int(np.argmax(fine_scores))])

    preview = PreprocessImage(crop_res=crop_res, rot_angle=best,
                              verbose=False).process(raw_frame,
                                                     normalize=False)
    return {
        "image_rot_deg": best % 360.0,
        "angles": np.concatenate([coarse, fine]),
        "scores": np.concatenate([coarse_scores, fine_scores]),
        "preview": preview,
    }


def fit_center(raw_frame, ref_psf, image_rot_deg, crop_res):
    """Brightest-pixel center, refined once by convolution with the
    reference PSF. Centers are in the rotated-frame coordinates that
    ``PreprocessImage`` uses."""
    proc = PreprocessImage(crop_res=crop_res, rot_angle=image_rot_deg,
                           verbose=False)
    crop = proc.process(raw_frame, normalize=False)
    proc.find_conv_center(crop, ref_psf)
    preview = proc.process(raw_frame, normalize=False)
    return {
        "crop_cx": int(proc.cen_x),
        "crop_cy": int(proc.cen_y),
        "preview": preview,
    }


def fit_flips(raw_frame, ref_psf, image_rot_deg, crop_cx, crop_cy, crop_res):
    """Score the four flip combinations; highest conv score wins.

    Discrimination comes from the pupil's asymmetric features (spiders,
    bad-actuator masks). The GUI shows the four previews for human
    confirmation — on the bench, flips were always obvious by eye.
    """
    scores = {}
    previews = {}
    for flip_x in (False, True):
        for flip_y in (False, True):
            proc = PreprocessImage(
                crop_res=crop_res, rot_angle=image_rot_deg,
                center_x=crop_cx, center_y=crop_cy,
                flip_horizontal=flip_x, flip_vertical=flip_y,
                verbose=False)
            crop = proc.process(raw_frame, normalize=False)
            scores[(flip_x, flip_y)] = alignment_score(crop, ref_psf)
            previews[(flip_x, flip_y)] = crop
    best = max(scores, key=scores.get)
    return {
        "flip_x": best[0],
        "flip_y": best[1],
        "scores": scores,
        "preview": previews[best],
    }


def probe_coefficients(n_modes, amplitude=1.0):
    """The calibration probe: a fixed mix of odd Zernikes.

    A flat-wavefront PSF is centro-symmetric — rotation is undetermined
    (mod anything) and flips are meaningless on a null PSF in a clean
    simulation. (The bench got away with null-PSF sweeps only because
    real non-common-path aberrations broke the symmetry.) Poking odd
    modes of different azimuthal order (coma + trefoil for a 10-mode
    basis) breaks both centro- and mirror-symmetry, making rotation,
    flips, and scale all identifiable from one probe frame.
    """
    coefficients = np.zeros(int(n_modes))
    if n_modes >= 9:
        coefficients[5] = 0.6 * amplitude   # Noll 7 (vertical coma)
        coefficients[8] = 0.4 * amplitude   # Noll 10 (oblique trefoil)
    elif n_modes >= 6:
        coefficients[5] = 0.6 * amplitude
        coefficients[0] = 0.3 * amplitude
    else:
        coefficients[0] = 0.5 * amplitude
    return coefficients


def fit_dm_scale(probe_crop, ideal_psf_fn, probe_coeffs, *,
                 scale_grid=None):
    """Estimate the DM actuation scale from the captured probe frame.

    Finds the ideal-sim amplitude whose rendering of the probe best
    matches the observed response — the automated equivalent of the
    bench's eyeball slider (which tuned 1e-6 -> 1.3e-6 -> 1.4e-6 across
    sessions). Pure comparison: no new exposures needed.

    ``probe_crop`` is the preprocessed (rotation/center/flips applied,
    normalized) probe frame; ``ideal_psf_fn(coefficients)`` returns the
    ideal-sim PSF for a coefficient vector.
    """
    if scale_grid is None:
        scale_grid = np.geomspace(0.4, 2.5, 21)

    def cosine(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    scores = []
    for scale in scale_grid:
        ideal_img = np.asarray(ideal_psf_fn(np.asarray(probe_coeffs) * scale))
        span = np.ptp(ideal_img)
        ideal_norm = (ideal_img - ideal_img.min()) / span if span > 0 else ideal_img
        scores.append(cosine(probe_crop, ideal_norm))
    scores = np.array(scores)
    best = float(scale_grid[int(np.argmax(scores))])
    return {
        "dm_scale": best,
        "scales": np.asarray(scale_grid, dtype=float),
        "scores": scores,
        "preview": probe_crop,
    }
