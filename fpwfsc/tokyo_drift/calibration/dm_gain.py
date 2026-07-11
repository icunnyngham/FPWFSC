"""Nominal command->wavefront gain of the bench DM.

The TranslationDM expresses a Zernike surface as per-actuator poke
commands; the DM's influence functions (overlapping actuator responses
with nearest-neighbour crosstalk) then render a smooth commanded surface
LARGER than the commanded poke amplitudes — ~1.5-1.7x for low-order
modes on the SCExAO 50x50 geometry. Any calibration that compares a
bench poke against an ideal-sim render at the *commanded* amplitude is
therefore comparing frames ~1.6x apart in aberration strength. A
non-coronagraphic PSF core survives that; a coronagraphic speckle field
decorrelates almost completely, which degrades the rotation/center fits
and biases the dm_scale fit toward ~1x.

The overshoot is a property of the NOMINAL DM model — actuator count,
pitch, influence-function shape: spec-sheet knowledge, not the injected
bench misalignments — so the calibration side may legitimately
precompute it and render its references at the matched amplitude. The
real bench absorbed the same physics empirically: dm_actuate_scale was
tuned to ~1.4e-6 against the 1e-6 nominal across the 2024 sessions.
"""
from functools import lru_cache

import numpy as np

from ..mode_registry import load_ts2_config, mode_n_modes
from .manual import probe_coefficients


@lru_cache(maxsize=None)
def nominal_dm_gain(mode_name):
    """Effective command->wavefront gain of the nominal bench DM for the
    mode's calibration-probe direction.

    Builds the unmisaligned bench DM (the same influence-function model
    ``derive_bench_config`` gives the bench sim) on the mode's pupil
    geometry, renders the standard calibration probe through the
    TranslationDM command path, projects the rendered surface back onto
    the mode's Zernike basis, and returns the amplitude ratio along the
    probe direction. Cached per mode (the build takes seconds).
    """
    import hcipy

    from ..dm import TranslationDM
    from ..sim.bench_sim import (
        DM_CROSSTALK,
        DM_INFLUENCE,
        DM_NOMINAL_SCALE,
        DM_NUM_ACTUATORS,
        DM_PITCH_M,
        MIN_PUPIL_EXTENT,
    )

    cfg = load_ts2_config(mode_name)
    corr = cfg["correctors"][cfg["corrector_chain"][0]]
    n_modes = mode_n_modes(mode_name)
    diameter = float(corr["zernike_diameter"])
    ideal_scale = float(corr.get("actuate_scale", 1.0e-6))

    resolution = int(cfg["pupil"]["resolution"])
    extent = max(float(cfg["pupil"]["extent"]), MIN_PUPIL_EXTENT)
    grid = hcipy.make_pupil_grid(resolution, extent)

    # The mode's Zernike basis, peak-normalized exactly as the ideal
    # sim's corrector builds it.
    basis = hcipy.make_zernike_basis(
        n_modes, diameter, grid,
        starting_mode=int(corr.get("starting_mode", 2)))
    basis = hcipy.ModeBasis([b / np.max(np.abs(b)) for b in basis])
    matrix = np.column_stack([np.asarray(b, float) for b in basis])
    mask = np.hypot(grid.x, grid.y) <= diameter / 2.0

    influence = {
        "gaussian": hcipy.make_gaussian_influence_functions,
        "xinetics": hcipy.make_xinetics_influence_functions,
    }[DM_INFLUENCE]
    kwargs = {"crosstalk": DM_CROSSTALK} if DM_INFLUENCE == "gaussian" else {}
    dm = hcipy.DeformableMirror(
        influence(grid, DM_NUM_ACTUATORS, DM_PITCH_M, **kwargs))

    probe = probe_coefficients(n_modes, amplitude=1.0)
    translator = TranslationDM(n_modes=n_modes,
                               dm_actuate_scale=DM_NOMINAL_SCALE,
                               zernike_diameter=diameter)
    dm.actuators = (translator.command_microns(probe)
                    .reshape(-1).astype(float) * 1e-6)

    surface = np.asarray(dm.surface, float)[mask]
    surface = surface - surface.mean()          # piston is unobservable
    fitted, *_ = np.linalg.lstsq(matrix[mask], surface, rcond=None)
    fitted /= ideal_scale                       # ideal-corrector units
    return float(np.dot(fitted, probe) / np.dot(probe, probe))
