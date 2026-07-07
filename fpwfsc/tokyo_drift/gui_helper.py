# gui_helper.py — tokyo_drift
"""GUI helper for the tokyo_drift pipeline.

Same contract as the other pipelines' gui_helper modules:
``valid_instruments`` populates the hardware dropdown, ``config_info``
carries per-field help/expert metadata for the auto-rendered config
form, and ``load_instruments`` maps a dropdown choice to a
``(Camera, AOSystem)`` pair.

tokyo_drift additions: ``list_modes`` / ``list_calibrations``
(re-exported from :mod:`.mode_registry`) populate the Mode and
Calibration dropdowns from the on-disk registries.
"""
from .mode_registry import (  # noqa: F401 — re-exported for GUI use
    CALIBRATIONS_DIR,
    MODES_DIR,
    PIPELINE_DIR,
    list_calibrations,
    list_modes,
)

valid_instruments = ['Sim', 'Vampires']


def _bench_sim_presets():
    from .sim.bench_sim import list_presets
    return list_presets()

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
        },
        "predictor": {
            "help": "Wavefront-error predictor: 'oracle' (sim-only, "
                    "reads injected truth - loop must converge) or "
                    "'random_walk' (noise - loop must diverge). The "
                    "trained NN predictor arrives with the model "
                    "integration milestone.",
            "expert": False
        },
        "strehl method": {
            "help": "Strehl estimator: 'vandam' (van Dam sub-pixel "
                    "peak + aperture photometry vs the pristine "
                    "reference) or 'proxy' (peak-to-total flux ratio)",
            "expert": True
        }
    },
    "CALIBRATION": {
        "probe amplitude": {
            "help": "Strength of the calibration probe poke (coma + "
                    "trefoil, in mode-coefficient units; ~1 um surface "
                    "peak per unit). Default 0.3 keeps the PSF "
                    "recognizable and near-linear; large values smear "
                    "it into speckle and inflate the residual floor.",
            "expert": True
        }
    },
    "CAMERA CALIBRATION": {
        "background file": {
            "help": "Path to background (dark) FITS file. Leave empty "
                    "to estimate from the frame border.",
            "file": True,
            "expert": True
        },
        "masterflat file": {
            "help": "Path to master flat FITS file. Leave empty to "
                    "skip flat fielding.",
            "file": True,
            "expert": True
        },
        "badpix file": {
            "help": "Path to bad pixel map FITS file. Leave empty to "
                    "skip bad pixel correction.",
            "file": True,
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
                    "bench: the priors for the camera rotation / crop "
                    "offset / DM rotation / scale / flips that "
                    "calibration must recover",
            "choices": _bench_sim_presets,
            "expert": False
        },
        "seed": {
            "help": "Random seed for the simulated bench. None = "
                    "fresh randomness each run.",
            "expert": True
        },
        "initial error rms": {
            "help": "Per-mode RMS of the hidden wavefront error "
                    "injected into the simulated bench - the "
                    "aberration the loop must correct.",
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
        },
        "hitchhiker mode": {
            "help": "Read frames from a watched directory of FITS "
                    "files instead of the camera (archive replay / "
                    "external image generators)",
            "expert": True
        },
        "hitchhiker path": {
            "help": "Directory the hitchhiker watches for new FITS "
                    "frames",
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


def get_choices(section, key):
    """Dropdown choices for a field, or None. A ``choices`` entry may be
    a list or a callable (evaluated at GUI build time, so registries
    scanned from disk stay fresh)."""
    choices = config_info.get(section, {}).get(key, {}).get("choices")
    if choices is None:
        return None
    return list(choices()) if callable(choices) else list(choices)


def is_expert_option(section, key):
    """Check if a given option is an expert option."""
    return config_info.get(section, {}).get(key, {}).get("expert", False)


def is_directory_option(section, key):
    """Check if a given option requires a directory selection."""
    return config_info.get(section, {}).get(key, {}).get("directory", False)


def is_file_option(section, key):
    """Check if a given option requires a file selection."""
    return config_info.get(section, {}).get(key, {}).get("file", False)
