#!/usr/bin/env python
"""tokyo_drift preflight — read-only deployment checks.

Run this after copying the repo to an instrument machine and BEFORE
grabbing the bench: it verifies the environment, the model checkpoints,
and that every hardware interface the loop will call actually exists —
without sending a single command.

    python -m fpwfsc.tokyo_drift.preflight [--config PATH] [--mode NAME]
                                           [--skip-hardware]

STRICTLY READ-ONLY: shared-memory streams are attached and read, never
written. No ``set_data``, no DM commands, no camera configuration.

Every check prints PASS / WARN / FAIL / SKIP; the exit code is nonzero
iff any check FAILs. Off-instrument (no pyMilk / no streams) the
hardware checks SKIP rather than fail, so the same command is useful on
the development machine.
"""
import argparse
import hashlib
import importlib
import inspect
import os
import sys
from pathlib import Path

# Must be set before torch imports (libomp double-load with hcipy/mkl).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PACKAGE_DIR = Path(__file__).parent

# md5 of the vendored model_torch.py, pinned in
# MODEL_INTEGRATION_NOTES.md (byte-identical copy of the training
# repo's pytorch_conversion module).
MODEL_TORCH_MD5 = "17174a7aab59f033e342c15dc6b4fe39"

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_COLORS = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m",
           SKIP: "\033[36m"}


class Report:
    def __init__(self, use_color=None):
        if use_color is None:
            use_color = sys.stdout.isatty()
        self.use_color = use_color
        self.results = []

    def add(self, status, name, detail=""):
        self.results.append((status, name, detail))
        tag = f"[{status}]"
        if self.use_color:
            tag = f"{_COLORS[status]}{tag}\033[0m"
        line = f"{tag:<15} {name}"
        if detail:
            line += f": {detail}"
        print(line)

    def section(self, title):
        print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))

    @property
    def failed(self):
        return any(s == FAIL for s, _, _ in self.results)

    def summary(self):
        counts = {s: 0 for s in (PASS, WARN, FAIL, SKIP)}
        for s, _, _ in self.results:
            counts[s] += 1
        print("\n" + "=" * 66)
        print("  ".join(f"{k}: {v}" for k, v in counts.items()))
        if self.failed:
            print("PREFLIGHT FAILED - fix the FAIL items before touching "
                  "the bench.")
        elif counts[WARN] or counts[SKIP]:
            print("Preflight passed with warnings/skips - read them before "
                  "proceeding.")
        else:
            print("Preflight clean.")


def _try_import(report, module, required=True, note=""):
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "")
        report.add(PASS, f"import {module}",
                   " ".join(x for x in (version, note) if x))
        return mod
    except Exception as e:
        report.add(FAIL if required else WARN, f"import {module}",
                   f"{type(e).__name__}: {e}" + (f" ({note})" if note else ""))
        return None


def check_environment(report):
    report.section("Environment")
    report.add(PASS, "python", f"{sys.version.split()[0]} "
               f"(env: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')})")
    if os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE":
        report.add(PASS, "KMP_DUPLICATE_LIB_OK", "TRUE (torch+hcipy libomp "
                   "coexistence)")
    _try_import(report, "numpy")
    _try_import(report, "hcipy")
    _try_import(report, "configobj")
    _try_import(report, "yaml")
    _try_import(report, "astropy")
    torch = _try_import(report, "torch")
    if torch is not None:
        report.add(PASS, "torch device", "cpu available"
                   + (", cuda available" if torch.cuda.is_available() else ""))
    _try_import(report, "PyQt5", required=False, note="GUI only")
    _try_import(report, "pyqtgraph", required=False, note="GUI plotter only")
    _try_import(report, "telescope_sim",
                note="needed on the bench too: renders the ideal "
                     "reference PSF and the calibration probe references")


def check_config(report, config_path, spec_path):
    report.section("Config")
    from ..common import support_functions as sf
    if not Path(config_path).exists():
        report.add(FAIL, "config file", f"{config_path} not found")
        return None
    try:
        settings = sf.validate_config(str(config_path), str(spec_path))
        report.add(PASS, "config validates", str(config_path))
        return settings
    except Exception as e:
        report.add(FAIL, "config validates", f"{type(e).__name__}: {e}")
        return None


def check_modes(report, mode_names):
    report.section("Modes / checkpoints")
    from .mode_registry import (checkpoint_path, list_calibrations,
                                list_modes, load_manifest, mode_n_modes)

    md5 = hashlib.md5((PACKAGE_DIR / "model_torch.py").read_bytes()).hexdigest()
    if md5 == MODEL_TORCH_MD5:
        report.add(PASS, "model_torch.py vendoring", f"md5 {md5}")
    else:
        report.add(FAIL, "model_torch.py vendoring",
                   f"md5 {md5} != pinned {MODEL_TORCH_MD5} - the vendored "
                   f"copy drifted from the training repo")

    if not mode_names:
        mode_names = list_modes()
    if not mode_names:
        report.add(FAIL, "modes", "no modes found in the registry")
        return

    for mode in mode_names:
        try:
            manifest = load_manifest(mode)
            n_modes = mode_n_modes(mode)
            report.add(PASS, f"mode {mode}",
                       f"filter {manifest.get('filter')!r}, "
                       f"{n_modes} modes")
        except Exception as e:
            report.add(FAIL, f"mode {mode}", f"{type(e).__name__}: {e}")
            continue
        try:
            ckpt = checkpoint_path(mode)
            report.add(PASS, f"mode {mode} checkpoint", str(ckpt))
        except Exception as e:
            report.add(FAIL, f"mode {mode} checkpoint", str(e))
            continue
        try:
            from .model_predictor import TorchPredictor
            predictor = TorchPredictor.from_mode(mode, device="cpu")
            report.add(PASS, f"mode {mode} model loads",
                       f"run {predictor.run_id}, "
                       f"{predictor.n_modes} modes, cpu")
        except Exception as e:
            report.add(FAIL, f"mode {mode} model loads",
                       f"{type(e).__name__}: {e}")
        profiles = list_calibrations(mode)
        if profiles:
            report.add(PASS, f"mode {mode} calibrations",
                       ", ".join(profiles))
        else:
            report.add(WARN, f"mode {mode} calibrations",
                       "no saved calibration profiles - fit one before "
                       "the loop will run")


def check_pymilk(report):
    """Verify the pyMilk API matches what the hardware classes call."""
    report.section("pyMilk API")
    try:
        from pyMilk.interfacing.isio_shmlib import SHM
    except Exception as e:
        report.add(SKIP, "pyMilk", f"not importable ({e}) - hardware "
                   f"checks will be skipped")
        return None
    report.add(PASS, "pyMilk import",
               "pyMilk.interfacing.isio_shmlib.SHM")

    params = set(inspect.signature(SHM.get_data).parameters)
    if {"check", "timeout"} <= params:
        report.add(PASS, "SHM.get_data signature",
                   "keyword args 'check'/'timeout' present")
    else:
        report.add(FAIL, "SHM.get_data signature",
                   f"expected 'check'/'timeout' kwargs, found {sorted(params)}"
                   f" - hardware classes call get_data(check=..., timeout=...)")
    if hasattr(SHM, "multi_recv_data"):
        report.add(PASS, "SHM.multi_recv_data", "present (frame averaging)")
    else:
        report.add(WARN, "SHM.multi_recv_data",
                   "absent - frame averaging falls back to a get_data loop")
    return SHM


def _read_stream(SHM, name):
    """Attach to a stream and read its current buffer. Read-only."""
    stream = SHM(name)
    data = stream.get_data(check=False)
    return stream, data


def check_hardware(report, SHM, dm_channel, mode_names,
                   camera_stream="vcam1", dark_stream="vcam1_dark",
                   injection_channel=""):
    report.section("Hardware interfaces (read-only)")
    if SHM is None:
        report.add(SKIP, "hardware", "pyMilk unavailable")
        return

    # Camera stream + filter keyword vs mode manifest
    filter_name = None
    try:
        cam, frame = _read_stream(SHM, camera_stream)
        report.add(PASS, f"camera stream {camera_stream!r}",
                   f"frame {frame.shape} {frame.dtype}")
        try:
            kwds = cam.get_keywords()
            filter_name = str(kwds.get("FILTER01", "")).strip()
            if filter_name:
                report.add(PASS, "camera FILTER01 keyword", filter_name)
            else:
                report.add(WARN, "camera FILTER01 keyword",
                           f"absent (keywords: {sorted(kwds)[:8]}...)")
        except Exception as e:
            report.add(WARN, "camera keywords", f"{type(e).__name__}: {e}")
    except Exception as e:
        report.add(SKIP, f"camera stream {camera_stream!r}",
                   f"not readable ({type(e).__name__}: {e})")

    if filter_name:
        from .run import _filter_number
        from .mode_registry import load_manifest
        for mode in mode_names:
            want = load_manifest(mode).get("filter")
            if not want:
                continue
            if _filter_number(want) == _filter_number(filter_name):
                report.add(PASS, f"filter matches mode {mode}",
                           f"{filter_name!r} ~ {want!r}")
            else:
                report.add(WARN, f"filter vs mode {mode}",
                           f"camera {filter_name!r} != trained {want!r} "
                           f"(fine if you will change filter or mode)")

    # Dark stream
    try:
        _, dark = _read_stream(SHM, dark_stream)
        report.add(PASS, f"dark stream {dark_stream!r}",
                   f"{dark.shape} {dark.dtype}")
    except Exception as e:
        report.add(WARN, f"dark stream {dark_stream!r}",
                   f"not readable ({type(e).__name__}) - write a dark "
                   f"before the run, or configure a background file")

    # DM channel: attach and READ. Never write.
    try:
        _, dm_data = _read_stream(SHM, dm_channel)
        detail = f"{dm_data.shape} {dm_data.dtype}"
        import numpy as np
        detail += (f", current rms {float(np.std(dm_data)):.3g}, "
                   f"max |{float(np.max(np.abs(dm_data))):.3g}|")
        if tuple(dm_data.shape) == (50, 50):
            report.add(PASS, f"DM channel {dm_channel!r}", detail)
        else:
            report.add(FAIL, f"DM channel {dm_channel!r}",
                       detail + " - expected (50, 50)")
    except Exception as e:
        report.add(SKIP, f"DM channel {dm_channel!r}",
                   f"not readable ({type(e).__name__}: {e})")

    # Injection channel ([SIMULATION] 'injection dm channel'): same
    # read-only attach as the correction channel, when configured.
    if injection_channel:
        try:
            _, inj_data = _read_stream(SHM, injection_channel)
            import numpy as np
            detail = (f"{inj_data.shape} {inj_data.dtype}, current rms "
                      f"{float(np.std(inj_data)):.3g}")
            if injection_channel == dm_channel:
                report.add(FAIL, f"injection channel {injection_channel!r}",
                           "same as the correction 'dm channel' - the "
                           "loop would overwrite its own injection")
            elif tuple(inj_data.shape) == (50, 50):
                report.add(PASS, f"injection channel {injection_channel!r}",
                           detail)
            else:
                report.add(FAIL, f"injection channel {injection_channel!r}",
                           detail + " - expected (50, 50)")
        except Exception as e:
            report.add(SKIP, f"injection channel {injection_channel!r}",
                       f"not readable ({type(e).__name__}: {e})")

    # Optional Subaru-only package
    try:
        import vampires_control  # noqa: F401
        report.add(PASS, "vampires_control", "importable (filter lookup)")
    except Exception:
        report.add(WARN, "vampires_control",
                   "not importable - Vampires wavelength lookup will "
                   "warn; the loop itself does not need it")

    # The classes the run will instantiate (camera read + DM attach only)
    try:
        from ..common import bench_hardware as hw
        report.add(PASS, "bench_hardware import", "Subaru classes present"
                   if hasattr(hw, "Vampires") else "?")
    except Exception as e:
        report.add(FAIL, "bench_hardware import",
                   f"{type(e).__name__}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only tokyo_drift deployment checks (no commands "
                    "are sent to any hardware).")
    parser.add_argument("--config",
                        default=str(PACKAGE_DIR / "tokyo_drift_config.ini"),
                        help="config .ini to validate (default: packaged)")
    parser.add_argument("--mode", action="append", default=[],
                        help="mode(s) to check (default: all registered)")
    parser.add_argument("--camera-stream", default="vcam1")
    parser.add_argument("--dark-stream", default="vcam1_dark")
    parser.add_argument("--skip-hardware", action="store_true",
                        help="skip shm stream checks even if pyMilk exists")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    report = Report(use_color=not args.no_color if args.no_color else None)
    print("tokyo_drift preflight - READ-ONLY: no commands will be sent.")

    check_environment(report)
    settings = check_config(report, args.config,
                            PACKAGE_DIR / "tokyo_drift_config.spec")
    check_modes(report, args.mode)

    dm_channel = "dm00disp04"
    injection_channel = ""
    if settings is not None:
        dm_channel = settings["DM"]["dm channel"]
        injection_channel = settings["SIMULATION"]["injection dm channel"]
    if args.skip_hardware:
        report.section("Hardware interfaces (read-only)")
        report.add(SKIP, "hardware", "--skip-hardware")
    else:
        SHM = check_pymilk(report)
        from .mode_registry import list_modes
        mode_names = args.mode or list_modes()
        check_hardware(report, SHM, dm_channel, mode_names,
                       camera_stream=args.camera_stream,
                       dark_stream=args.dark_stream,
                       injection_channel=injection_channel)

    report.summary()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
