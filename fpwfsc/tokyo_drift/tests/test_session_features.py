"""Session logging, hitchhiker usage pattern, and Strehl estimators."""
import json

import numpy as np
import pytest

from fpwfsc.tokyo_drift.session_log import SessionLogger


def test_session_logger_layout(tmp_path):
    logger = SessionLogger(tmp_path, settings={"MODE": {"mode name": "m"}},
                           session_name="test_session")
    assert (logger.session_dir / "config.json").is_file()

    logger.save_iteration(0, strehl=0.5, state=np.zeros(10),
                          prediction=np.ones(10),
                          dm_command=np.full((50, 50), 0.1),
                          raw=np.zeros((256, 256)),
                          processed=np.zeros((128, 128)))
    iter_dir = logger.session_dir / "iter_000"
    with open(iter_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["strehl"] == 0.5
    assert len(meta["state"]) == 10
    assert meta["dm_command_rms_um"] == pytest.approx(0.1)
    for name in ("raw", "processed", "dm_command"):
        assert (iter_dir / f"{name}.fits").is_file()

    logger.finalize({"iterations": 1, "strehls": [0.5],
                     "final_state": np.zeros(10)})
    with open(logger.session_dir / "summary.json") as f:
        summary = json.load(f)
    assert summary["iterations"] == 1


def test_session_logger_nan_strehl_serializes(tmp_path):
    logger = SessionLogger(tmp_path, session_name="nan_session")
    logger.save_iteration(0, strehl=float("nan"), state=np.zeros(3))
    with open(logger.session_dir / "iter_000" / "metadata.json") as f:
        assert json.load(f)["strehl"] is None


def test_sessions_do_not_overwrite(tmp_path):
    a = SessionLogger(tmp_path, session_name="s1")
    b = SessionLogger(tmp_path, session_name="s2")
    assert a.session_dir != b.session_dir


def test_hitchhiker_usage_pattern(tmp_path):
    """Pin the exact way run.py drives common/'s Hitchhiker."""
    fits = pytest.importorskip("astropy.io.fits")
    from fpwfsc.common.fake_hardware import Hitchhiker

    hitch = Hitchhiker(imagedir=tmp_path, poll_interval=0.05, timeout=5)
    frame = np.arange(64.0).reshape(8, 8)
    fits.writeto(tmp_path / "frame_0001.fits", frame)
    received = hitch.wait_for_next_image()
    np.testing.assert_array_equal(received, frame)


def test_vandam_strehl_sanity():
    from fpwfsc.common import vandamstrehl as vd

    yy, xx = np.mgrid[-64:64, -64:64]
    ref = np.exp(-(xx**2 + yy**2) / (2 * 3.0**2))
    blurred = np.exp(-(xx**2 + yy**2) / (2 * 6.0**2))
    assert vd.strehl(ref.copy(), ref.copy()) == pytest.approx(1.0, abs=0.02)
    assert vd.strehl(blurred, ref.copy()) < 0.5
