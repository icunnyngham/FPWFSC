"""Per-run session logging.

Adapts the FPWFSC ``LogManager`` pattern (per-iteration directory of
FITS arrays + ``metadata.json``, config snapshot saved once) with one
fix: every run gets its own timestamped session directory, so
successive runs never overwrite each other (LogManager writes
``iter_NNN`` straight into its base directory).

Layout::

    <log_path>/tokyo_drift_<YYYY-mm-ddTHH-MM-SS>/
    ├── config.json            validated settings snapshot
    ├── calibration_profile.yaml  copy of the profile the run used
    ├── background.fits        background/dark the reducer subtracted
    ├── camera_state.json      camera-side state at run start (filter,
    │                          shm keywords, dark info, filter-vs-mode
    │                          comparison; hardware only)
    ├── episode.json           episode index, injected error coeffs
    │                          (sim, or hardware injection), initial
    │                          diversity move
    ├── injected_command.fits  50x50 microns injection actually written
    │                          to the injection DM channel (hardware
    │                          injection runs only)
    ├── iter_000/
    │   ├── metadata.json      iteration, strehl, state, prediction, rms
    │   ├── raw.fits           frame as acquired (post-reduction)
    │   ├── processed.fits     preprocessed crop the predictor saw
    │   ├── camera_raw.fits    pre-reduction readout cube, native dtype
    │   │                      ([IO] 'save camera frames', hardware only)
    │   ├── dm_command.fits    50x50 microns command sent
    │   └── dm_command_refused.fits  command REFUSED by the DM safety
    │                          bounds (never sent; only on the aborting
    │                          iteration, which has no dm_command.fits)
    ├── iter_001/ ...
    └── summary.json           strehl history, final state, iterations,
                               aborted / safety_error on a safety abort

This is the offline record for "why did this run diverge": every input
and output of every iteration, replayable against the saved config. The
profile copy and background frame make the directory self-describing —
config.json records the profile only by path, which for GUI
run-with-unsaved-parameters sessions is an ephemeral tempfile.
"""
import datetime
import json
import os
import shutil
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

    def save_provenance(self, profile_path=None, background=None,
                        camera_state=None):
        """Copy the run's calibration inputs into the session directory.

        ``profile_path`` is copied to ``calibration_profile.yaml`` (the
        original may be an ephemeral tempfile, or edited later);
        ``background`` is the frame the reducer actually subtracts — on
        hardware the camera dark, whose shm buffer is overwritten by the
        next dark taken. ``camera_state`` is the camera-side snapshot
        (filter, shm keywords, ...) config.json cannot reconstruct.
        """
        if profile_path is not None:
            shutil.copyfile(profile_path,
                            self.session_dir / "calibration_profile.yaml")
        if background is not None:
            fits.writeto(self.session_dir / "background.fits",
                         np.asarray(background, dtype=float),
                         overwrite=True)
        if camera_state is not None:
            with open(self.session_dir / "camera_state.json", "w") as f:
                json.dump(camera_state, f, indent=2, default=str)

    def save_episode(self, *, episode=0, n_repeats=1,
                     injected_error_coeffs=None, initial_move=None,
                     injected_command=None):
        """Record the episode-level context per-iteration metadata can't
        carry: without the injected error (sim, or hardware injection)
        and the initial diversity move, ``state[0]`` is not decomposable
        offline. ``injected_command`` is the 50x50 microns command a
        hardware injection run actually wrote to the injection channel
        (only present when one was applied)."""
        episode_info = {
            "episode": int(episode),
            "n_repeats": int(n_repeats),
            "injected_error_coeffs": None if injected_error_coeffs is None
            else np.asarray(injected_error_coeffs).tolist(),
            "initial_move": None if initial_move is None
            else np.asarray(initial_move).tolist(),
        }
        with open(self.session_dir / "episode.json", "w") as f:
            json.dump(episode_info, f, indent=2)
        if injected_command is not None:
            fits.writeto(self.session_dir / "injected_command.fits",
                         np.asarray(injected_command, dtype=float),
                         overwrite=True)

    def save_iteration(self, iteration, *, strehl=None, state=None,
                       prediction=None, dm_command=None, raw=None,
                       processed=None, camera_frames=None,
                       dm_command_refused=None, safety_error=None):
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
            "command_sent": dm_command_refused is None,
            "safety_error": safety_error,
        }
        with open(iter_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        for name, array in (("raw", raw), ("processed", processed),
                            ("dm_command", dm_command),
                            ("dm_command_refused", dm_command_refused)):
            if array is not None:
                fits.writeto(iter_dir / f"{name}.fits",
                             np.asarray(array, dtype=float), overwrite=True)
        if camera_frames is not None:
            # Native dtype on purpose: this is the pre-reduction readout
            # cube (typically uint16), kept exactly as the camera
            # delivered it.
            fits.writeto(iter_dir / "camera_raw.fits",
                         np.asarray(camera_frames), overwrite=True)

    def finalize(self, result):
        """Write the run summary (strehl history, final state)."""
        summary = {
            "iterations": int(result.get("iterations", 0)),
            "strehls": np.asarray(result.get("strehls", []),
                                  dtype=float).tolist(),
            "final_state": np.asarray(result.get("final_state", []),
                                      dtype=float).tolist(),
            "aborted": result.get("aborted"),
            "safety_error": result.get("safety_error"),
        }
        with open(self.session_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return self.session_dir
