"""Smoke test for the closed-loop timing benchmark."""
from pathlib import Path

import pytest
from configobj import ConfigObj

PIPELINE_DIR = Path(__file__).resolve().parents[1]
SPEC = str(PIPELINE_DIR / "tokyo_drift_config.spec")
SIM_INI = str(PIPELINE_DIR / "tokyo_drift_config_sim.ini")


def test_benchmark_oracle_smoke(capsys):
    """The benchmark builds the real loop, times every seam, and the
    stage accounting is consistent. Oracle predictor: checkpoint-free."""
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.benchmark import print_report, run_benchmark

    cfg = ConfigObj(SIM_INI)
    cfg["LOOP_SETTINGS"]["predictor"] = "oracle"
    cfg["SIMULATION"]["wfe seed"] = "27"
    cfg["SNR"]["frames to average"] = "2"

    result = run_benchmark(cfg, SPEC, n_iter=2)

    assert result["n_iter"] == 2
    assert len(result["residuals"]) == 2
    # One frame before the loop + one per iteration
    assert result["stages"]["camera total (render+mangle+noise)"][1] == 3
    assert result["stages"]["camera: TS2 render"][1] == 3
    assert result["stages"]["NN inference"][1] == 2
    assert result["stages"]["command translation"][1] == 2
    # The render is a sub-timer of the camera total
    assert (result["stages"]["camera: TS2 render"][0]
            <= result["stages"]["camera total (render+mangle+noise)"][0])
    # Stage time is accounted within the loop wall
    accounted = (result["stages"]["camera total (render+mangle+noise)"][0]
                 + sum(secs for name, (secs, _) in result["stages"].items()
                       if not name.startswith("camera")))
    assert accounted <= result["loop_wall"]
    # No profile configured -> calibration ran as a one-time setup cost
    assert "calibration (no profile configured)" in result["setup"]

    print_report(result)
    out = capsys.readouterr().out
    assert "per-stage" in out and "TS2 render" in out
