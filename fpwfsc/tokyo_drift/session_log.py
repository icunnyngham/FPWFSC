"""Per-run session logging.

Adapts the FPWFSC ``LogManager`` pattern (per-iteration directory of
FITS arrays + ``metadata.json``, config snapshot saved once) with one
fix: every run gets its own timestamped session directory, so
successive runs never overwrite each other (LogManager writes
``iter_NNN`` straight into its base directory).

Layout::

    <log_path>/tokyo_drift_<YYYY-mm-ddTHH-MM-SS>/
    ├── config.json            validated settings snapshot
    ├── iter_000/
    │   ├── metadata.json      iteration, strehl, state, prediction, rms
    │   ├── raw.fits           frame as acquired (post-reduction)
    │   ├── processed.fits     preprocessed crop the predictor saw
    │   └── dm_command.fits    50x50 microns command sent
    ├── iter_001/ ...
    └── summary.json           strehl history, final state, iterations

This is the offline record for "why did this run diverge": every input
and output of every iteration, replayable against the saved config.
"""
import datetime
import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits


class SessionLogger:
    """Write one session directory per run under ``base_log_dir``."""

    def __init__(self, base_log_dir, settings=None, session_name=None):
        stamp = session_name or datetime.datetime.now().strftime(
            "tokyo_drift_%Y-%m-%dT%H-%M-%S")
        self.session_dir = Path(base_log_dir or ".") / stamp
        os.makedirs(self.session_dir, exist_ok=True)

        if settings is not None:
            snapshot = settings.dict() if hasattr(settings, "dict") else dict(settings)
            with open(self.session_dir / "config.json", "w") as f:
                json.dump(snapshot, f, indent=2, default=str)

    def save_iteration(self, iteration, *, strehl=None, state=None,
                       prediction=None, dm_command=None, raw=None,
                       processed=None):
        iter_dir = self.session_dir / f"iter_{iteration:03d}"
        os.makedirs(iter_dir, exist_ok=True)

        meta = {
            "iteration": int(iteration),
            "strehl": None if strehl is None or not np.isfinite(strehl)
                      else float(strehl),
            "state": None if state is None else np.asarray(state).tolist(),
            "prediction": None if prediction is None
                          else np.asarray(prediction).tolist(),
            "dm_command_rms_um": None if dm_command is None
                                 else float(np.sqrt(np.mean(
                                     np.square(dm_command)))),
        }
        with open(iter_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        for name, array in (("raw", raw), ("processed", processed),
                            ("dm_command", dm_command)):
            if array is not None:
                fits.writeto(iter_dir / f"{name}.fits",
                             np.asarray(array, dtype=float), overwrite=True)

    def finalize(self, result):
        """Write the run summary (strehl history, final state)."""
        summary = {
            "iterations": int(result.get("iterations", 0)),
            "strehls": np.asarray(result.get("strehls", []),
                                  dtype=float).tolist(),
            "final_state": np.asarray(result.get("final_state", []),
                                      dtype=float).tolist(),
        }
        with open(self.session_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return self.session_dir
