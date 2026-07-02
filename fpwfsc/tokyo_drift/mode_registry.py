"""Mode and calibration registries for the tokyo_drift pipeline.

A *mode* is a trained-model setup: a directory under ``modes/`` pairing
a telescope-sim (TS2) optical configuration with an NN checkpoint
pointer and descriptive metadata:

    modes/<mode_name>/
    ├── ts2_config.yaml   TS2 config the model was trained against
    └── manifest.yaml     instrument / filter / basis / checkpoint pointer

A *calibration profile* is a ``calibrations/<name>.yaml`` file holding
the fitted instrument-alignment parameters (rotation, center, flips,
DM scale) for a specific mode; profiles carry a ``mode:`` key as the
foreign key and are filtered by it.
"""
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
MODES_DIR = PIPELINE_DIR / "modes"
CALIBRATIONS_DIR = PIPELINE_DIR / "calibrations"


def list_modes(modes_dir=MODES_DIR):
    """Names of registered modes (subdirectories of ``modes/``)."""
    if not Path(modes_dir).is_dir():
        return []
    return sorted(p.name for p in Path(modes_dir).iterdir()
                  if p.is_dir() and not p.name.startswith(('_', '.')))


def list_calibrations(mode_name=None, calibrations_dir=CALIBRATIONS_DIR):
    """Names of saved calibration profiles, optionally filtered by mode."""
    if not Path(calibrations_dir).is_dir():
        return []
    try:
        import yaml
    except ImportError:
        return []
    profiles = []
    for path in sorted(Path(calibrations_dir).glob("*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"Skipping unreadable calibration profile {path.name}: {exc}")
            continue
        if mode_name is None or data.get("mode") == mode_name:
            profiles.append(path.stem)
    return profiles


def mode_dir(mode_name, modes_dir=MODES_DIR):
    """Directory of a registered mode; raises KeyError if unknown."""
    path = Path(modes_dir) / mode_name
    if not path.is_dir():
        raise KeyError(
            f"unknown mode {mode_name!r}; registered modes: "
            f"{list_modes(modes_dir)}")
    return path


def ts2_config_path(mode_name, modes_dir=MODES_DIR):
    """Path to the mode's telescope-sim YAML config."""
    path = mode_dir(mode_name, modes_dir) / "ts2_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"mode {mode_name!r} has no ts2_config.yaml")
    return path


def load_manifest(mode_name, modes_dir=MODES_DIR):
    """The mode's manifest.yaml as a dict ({} if absent)."""
    path = mode_dir(mode_name, modes_dir) / "manifest.yaml"
    if not path.is_file():
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}
