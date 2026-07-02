"""Entry-point + config-contract tests for the tokyo_drift pipeline."""
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from configobj import ConfigObj

from fpwfsc.common import support_functions as sf
from fpwfsc.tokyo_drift.run import run

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


def test_invalid_config_raises(tmp_path):
    cfg = ConfigObj(SIM_INI)
    cfg["LOOP_SETTINGS"]["N iter"] = 0  # violates integer(min=1)
    bad = tmp_path / "bad.ini"
    cfg.filename = str(bad)
    cfg.write()
    with pytest.raises(ValueError):
        sf.validate_config(str(bad), SPEC)


def test_run_sim_returns_validated_settings():
    settings = run("Sim", "Sim", config=SIM_INI, configspec=SPEC)
    assert settings["MODE"]["mode name"] == "vampires_f760_10zern"
    assert settings["LOOP_SETTINGS"]["N iter"] >= 1


def test_run_real_hardware_not_implemented():
    with pytest.raises(NotImplementedError):
        run(object(), object(), config=SIM_INI, configspec=SPEC)


def test_run_respects_stop_event():
    stop = threading.Event()
    stop.set()
    settings = run("Sim", "Sim", config=SIM_INI, configspec=SPEC,
                   my_event=stop)
    assert settings is not None


def test_module_entry_point_runs_clean():
    result = subprocess.run(
        [sys.executable, "-m", "fpwfsc.tokyo_drift.run"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "config OK" in result.stdout
