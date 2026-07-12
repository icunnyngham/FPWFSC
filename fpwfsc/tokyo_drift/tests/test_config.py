"""Entry-point + config-contract tests for the tokyo_drift pipeline."""
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from configobj import ConfigObj

from fpwfsc.common import support_functions as sf
from fpwfsc.tokyo_drift.run import make_frame_reducer, run

PIPELINE_DIR = Path(__file__).resolve().parents[1]
SPEC = str(PIPELINE_DIR / "tokyo_drift_config.spec")
BENCH_INI = str(PIPELINE_DIR / "tokyo_drift_config.ini")
SIM_INI = str(PIPELINE_DIR / "tokyo_drift_config_sim.ini")


@pytest.mark.parametrize("ini", [BENCH_INI, SIM_INI], ids=["bench", "sim"])
def test_shipped_configs_validate(ini):
    settings = sf.validate_config(ini, SPEC)
    for section in ("MODE", "LOOP_SETTINGS", "DM", "SIMULATION", "IO"):
        assert section in settings


def test_spec_coerces_types():
    settings = sf.validate_config(SIM_INI, SPEC)
    assert isinstance(settings["LOOP_SETTINGS"]["N iter"], int)
    assert isinstance(settings["LOOP_SETTINGS"]["gain"], float)
    assert isinstance(settings["IO"]["save_log"], bool)
    assert settings["MODE"]["calibration profile"] is None
    assert settings["LOOP_SETTINGS"]["strehl early stop"] is None
    assert settings["SIMULATION"]["seed"] == 27
    # SNR section: sim source brightness (10^x) + frame averaging (used
    # by both the loop and calibration; default 8 so the coronagraph
    # modes work out of the box at the training-era flux)
    assert settings["SNR"]["int phot flux exponent"] == 3.5
    assert settings["SNR"]["frames to average"] == 8


def test_border_background_estimation_defaults_off(tmp_path):
    # The field must default False even for configs written before it
    # existed (the spec default fills it in).
    cfg = ConfigObj(SIM_INI)
    del cfg["CAMERA CALIBRATION"]["estimate background from border"]
    old = tmp_path / "pre_field.ini"
    cfg.filename = str(old)
    cfg.write()
    settings = sf.validate_config(str(old), SPEC)
    assert settings["CAMERA CALIBRATION"][
        "estimate background from border"] is False


def test_frame_reducer_border_estimation_toggle():
    rng = np.random.default_rng(0)
    frame = rng.uniform(10.0, 20.0, (32, 32))
    no_files = {"bkgd": None, "masterflat": None, "badpix": None}

    # Off (default): no background file -> frame passes through untouched.
    reduce_off = make_frame_reducer(no_files, False)
    np.testing.assert_array_equal(reduce_off(frame), frame)

    # On: matches equalize_image's border-median fallback.
    reduce_on = make_frame_reducer(no_files, True)
    np.testing.assert_array_equal(reduce_on(frame),
                                  sf.equalize_image(frame))

    # A configured background frame is used regardless of the toggle.
    bkgd = np.full_like(frame, 5.0)
    reduce_meas = make_frame_reducer({**no_files, "bkgd": bkgd}, False)
    np.testing.assert_allclose(reduce_meas(frame), frame - 5.0)

    # Shape mismatch still fails loudly.
    with pytest.raises(ValueError, match="shape"):
        make_frame_reducer({**no_files, "bkgd": np.zeros((8, 8))},
                           False)(frame)


def test_invalid_config_raises(tmp_path):
    cfg = ConfigObj(SIM_INI)
    cfg["LOOP_SETTINGS"]["N iter"] = 0  # violates integer(min=1)
    bad = tmp_path / "bad.ini"
    cfg.filename = str(bad)
    cfg.write()
    with pytest.raises(ValueError):
        sf.validate_config(str(bad), SPEC)


def test_run_sim_returns_validated_settings():
    stop = threading.Event()
    stop.set()  # validate the config contract without building backends
    result = run("Sim", "Sim", config=SIM_INI, configspec=SPEC,
                 my_event=stop)
    settings = result["settings"]
    assert settings["MODE"]["mode name"] == "vampires_f760_10zern"
    assert settings["LOOP_SETTINGS"]["N iter"] >= 1
    assert result["loop"] is None


def test_run_real_hardware_not_implemented():
    with pytest.raises(NotImplementedError):
        run(object(), object(), config=SIM_INI, configspec=SPEC)


def test_run_respects_stop_event():
    stop = threading.Event()
    stop.set()
    result = run("Sim", "Sim", config=SIM_INI, configspec=SPEC,
                 my_event=stop)
    assert result["loop"] is None


def test_module_entry_point_check_config():
    result = subprocess.run(
        [sys.executable, "-m", "fpwfsc.tokyo_drift.run", "--check-config"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "config OK" in result.stdout
