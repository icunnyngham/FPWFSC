"""Preflight CLI tests (the read-only bench-deployment checker)."""
import numpy as np
import pytest

from fpwfsc.tokyo_drift import preflight


def test_report_fails_iff_any_fail():
    r = preflight.Report(use_color=False)
    r.add(preflight.PASS, "a")
    r.add(preflight.WARN, "b")
    r.add(preflight.SKIP, "c")
    assert not r.failed
    r.add(preflight.FAIL, "d")
    assert r.failed


def test_filter_number_tolerates_naming_conventions():
    from fpwfsc.tokyo_drift.run import _filter_number
    assert _filter_number("F750") == "750"
    assert _filter_number("750-50") == "750"
    assert _filter_number("Broad") is None
    assert _filter_number("F750") == _filter_number("750-50")


def test_model_torch_md5_pin_matches_vendored_file():
    """The constant in preflight must track the vendored file (it is the
    same pin as MODEL_INTEGRATION_NOTES.md)."""
    import hashlib
    md5 = hashlib.md5(
        (preflight.PACKAGE_DIR / "model_torch.py").read_bytes()).hexdigest()
    assert md5 == preflight.MODEL_TORCH_MD5


def test_main_off_instrument_passes(capsys):
    """End-to-end with hardware skipped: on a machine with the dev env
    and checkpoints, preflight must exit clean (0)."""
    pytest.importorskip("torch")
    from fpwfsc.tokyo_drift.mode_registry import checkpoint_path
    try:
        checkpoint_path("vampires_f760_10zern")
    except Exception:
        pytest.skip("checkpoints not present on this machine")
    rc = preflight.main(["--skip-hardware", "--no-color",
                         "--mode", "vampires_f760_10zern"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "READ-ONLY" in out
    assert "FAIL: 0" in out


def test_main_fails_on_bad_config(tmp_path, capsys):
    pytest.importorskip("torch")
    bad = tmp_path / "bad.ini"
    bad.write_text("[MODE]\n")  # missing required fields
    rc = preflight.main(["--skip-hardware", "--no-color",
                         "--config", str(bad),
                         "--mode", "vampires_f760_10zern"])
    assert rc == 1
