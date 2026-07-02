# gui_helper.py — tokyo_drift
"""GUI helper for the tokyo_drift pipeline.

Same contract as the other pipelines' gui_helper modules:
``valid_instruments`` populates the hardware dropdown, ``config_info``
carries per-field help/expert metadata for the auto-rendered config
form, and ``load_instruments`` maps a dropdown choice to a
``(Camera, AOSystem)`` pair.

tokyo_drift additions: ``list_modes`` / ``list_calibrations`` populate
the Mode and Calibration dropdowns from the on-disk registries
(``modes/<name>/`` directories and ``calibrations/<name>.yaml``
profiles; calibration profiles are filtered to the selected mode via
their ``mode:`` key).
"""
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
MODES_DIR = PIPELINE_DIR / "modes"
CALIBRATIONS_DIR = PIPELINE_DIR / "calibrations"

valid_instruments = ['Sim', 'Vampires']


def list_modes(modes_dir=MODES_DIR):
    """Names of registered modes (subdirectories of ``modes/``)."""
    if not Path(modes_dir).is_dir():
        return []
    return sorted(p.name for p in Path(modes_dir).iterdir()
                  if p.is_dir() and not p.name.startswith(('_', '.')))


def list_calibrations(mode_name=None, calibrations_dir=CALIBRATIONS_DIR):
    """Names of saved calibration profiles, optionally filtered by mode.

    Profiles are ``calibrations/<name>.yaml`` files carrying a
    ``mode: <mode_name>`` key; passing ``mode_name`` returns only the
    profiles fit against that mode.
    """
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

config_info = {
    "MODE": {
        "mode name": {
            "help": "Trained-model mode to run (a directory under "
                    "tokyo_drift/modes/ pairing a telescope-sim config "
                    "with an NN checkpoint)",
            "expert": False
        },
        "calibration profile": {
            "help": "Saved calibration profile to apply (rotation, "
                    "center, flips, DM scale). None = uncalibrated.",
            "expert": False
        }
    },
    "LOOP_SETTINGS": {
        "Plot": {
            "help": "Show the live plotter (PSF alignment panels, "
                    "Strehl history, mode coefficients)",
            "expert": False
        },
        "N iter": {
            "help": "Number of iterations for the loop",
            "expert": False
        },
        "gain": {
            "help": "Gain factor applied to each NN prediction",
            "expert": False
        },
        "leak factor": {
            "help": "Leak on the accumulated DM state (1.0 = no leak)",
            "expert": True
        },
        "strehl early stop": {
            "help": "Stop the loop early once the measured Strehl "
                    "exceeds this value. None disables early stop.",
            "expert": True
        }
    },
    "DM": {
        "dm channel": {
            "help": "MILK shared-memory DM channel to write "
                    "(e.g. dm00disp04)",
            "expert": True
        },
        "max peak to valley (um)": {
            "help": "Refuse DM commands whose surface peak-to-valley "
                    "exceeds this bound",
            "expert": True
        },
        "max actuator stroke (um)": {
            "help": "Refuse DM commands with any single actuator "
                    "beyond this stroke",
            "expert": True
        }
    },
    "SIMULATION": {
        "bench sim preset": {
            "help": "Misalignment-injection preset for the simulated "
                    "bench (easy, realistic_vampires, stress_test, ...)",
            "expert": False
        },
        "seed": {
            "help": "Random seed for the simulated bench. None = "
                    "fresh randomness each run.",
            "expert": True
        }
    },
    "IO": {
        "save_log": {
            "help": "Save per-iteration images, DM commands, "
                    "predictions, and metrics",
            "expert": False
        },
        "log_path": {
            "help": "Directory for per-run log folders. Empty = "
                    "current directory.",
            "directory": True,
            "expert": True
        }
    }
}


def load_instruments(instrumentname, camargs={}, aoargs={}):
    if instrumentname == 'Sim':
        return 'Sim', 'Sim'
    elif instrumentname == 'Vampires':
        # Imported lazily: bench_hardware needs SCExAO-only packages
        # that are absent when running sim mode off-instrument.
        from ..common import bench_hardware as hw
        return hw.Vampires(**camargs), hw.SCEXAO(**aoargs)
    else:
        raise ValueError(f"Invalid instrument name: {instrumentname!r}")


def get_help_message(section, key):
    """Retrieve the help message for a given section and key."""
    return config_info.get(section, {}).get(key, {}).get("help", "No help available")


def is_expert_option(section, key):
    """Check if a given option is an expert option."""
    return config_info.get(section, {}).get(key, {}).get("expert", False)


def is_directory_option(section, key):
    """Check if a given option requires a directory selection."""
    return config_info.get(section, {}).get(key, {}).get("directory", False)


def is_file_option(section, key):
    """Check if a given option requires a file selection."""
    return config_info.get(section, {}).get(key, {}).get("file", False)
