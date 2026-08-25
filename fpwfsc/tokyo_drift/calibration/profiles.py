"""Calibration-profile persistence.

A profile is one fitted instrument alignment for one mode:

    calibrations/<name>.yaml
        mode: vampires_f760_10zern     # foreign key, must match the run
        image_rot_deg: 235.4           # rotation APPLIED to raw frames
        crop_cx: 270                   # PSF center in the rotated frame
        crop_cy: 268                   #   (null -> auto brightest-pixel)
        flip_x: false                  # image flips (after crop)
        flip_y: false
        dm_scale: 1.3                  # actuation-scale factor vs nominal
        dm_rot_deg: 0.0                # DM rotation baked into the basis
        dm_flip_x: false               # DM command flips (Translation DM)
        dm_flip_y: false
        shift_x: 0                     # bench-only integer command shifts
        shift_y: 0
        saved_at: 2026-07-02T14:23:00

``shift_x``/``shift_y`` reproduce a bench-hardware artifact (the DM
pupil sitting off-center on the actuator grid). The simulated bench has
no such mechanism, so sim mode refuses profiles that set them —
``assert_sim_safe`` fails loudly rather than silently mis-modeling.
"""
import datetime
from pathlib import Path

from ..mode_registry import CALIBRATIONS_DIR

PROFILE_DEFAULTS = {
    "image_rot_deg": 0.0,
    "crop_cx": None,
    "crop_cy": None,
    "flip_x": False,
    "flip_y": False,
    "dm_scale": 1.0,
    "dm_rot_deg": 0.0,
    "dm_flip_x": False,
    "dm_flip_y": False,
    "shift_x": 0,
    "shift_y": 0,
}


def save_profile(name, profile, calibrations_dir=CALIBRATIONS_DIR):
    """Write a profile as ``<calibrations_dir>/<name>.yaml`` and return
    the path. Adds a ``saved_at`` timestamp."""
    import yaml
    out = dict(profile)
    out.setdefault("saved_at", datetime.datetime.now().isoformat(
        timespec="seconds"))
    path = Path(calibrations_dir) / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    return path


def resolve_profile_path(name_or_path, calibrations_dir=CALIBRATIONS_DIR):
    """The YAML file behind a profile reference: a direct path passes
    through, a registry name resolves under ``calibrations_dir``. Also
    what the session logger copies for provenance — the config may hold
    either form."""
    candidate = Path(str(name_or_path))
    if candidate.suffix == ".yaml" and candidate.is_file():
        return candidate
    path = Path(calibrations_dir) / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown calibration profile {name_or_path!r} "
            f"(no {path})")
    return path


def load_profile(name_or_path, calibrations_dir=CALIBRATIONS_DIR):
    """Load a profile by registry name, or by direct path to a YAML
    file (scripted runs / tests). Missing fields get defaults."""
    import yaml
    path = resolve_profile_path(name_or_path, calibrations_dir)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    profile = dict(PROFILE_DEFAULTS)
    profile.update(data)
    return profile


def delete_profile(name, calibrations_dir=CALIBRATIONS_DIR):
    """Delete a saved profile by registry name; return the removed path.
    Raises FileNotFoundError if it does not exist."""
    path = Path(calibrations_dir) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown calibration profile {name!r} (no {path})")
    path.unlink()
    return path


def assert_sim_safe(profile):
    """Refuse bench-only shift fields when running against the
    simulated bench (which has no shift mechanism)."""
    if profile.get("shift_x") or profile.get("shift_y"):
        raise ValueError(
            "calibration profile carries nonzero shift_x/shift_y - these "
            "reproduce a bench-hardware artifact the simulated bench does "
            "not model; fit a sim profile (shifts 0) instead")
