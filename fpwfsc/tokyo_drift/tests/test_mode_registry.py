"""Mode-registry and vendored-pupil tests."""
import numpy as np
import pytest

from fpwfsc.tokyo_drift import mode_registry as reg


def test_vampires_f760_mode_is_registered():
    assert "vampires_f760_10zern" in reg.list_modes()


def test_ts2_config_path_resolves():
    path = reg.ts2_config_path("vampires_f760_10zern")
    assert path.is_file()


def test_manifest_loads():
    manifest = reg.load_manifest("vampires_f760_10zern")
    assert manifest["mode"] == "vampires_f760_10zern"
    assert manifest["instrument"] == "Vampires"
    # A checkpoint pointer is configured (the file itself is gitignored).
    assert manifest["checkpoint"]


def test_checkpoint_path_errors_when_pointer_missing(tmp_path):
    """A mode whose manifest has no checkpoint gives a clear ValueError."""
    mode = tmp_path / "no_ckpt_mode"
    mode.mkdir()
    (mode / "manifest.yaml").write_text("mode: no_ckpt_mode\n")
    with pytest.raises(ValueError, match="no checkpoint"):
        reg.checkpoint_path("no_ckpt_mode", modes_dir=tmp_path)


def test_unknown_mode_raises():
    with pytest.raises(KeyError):
        reg.mode_dir("no_such_mode")


def test_vendored_pupil_matches_training_generator():
    """The vendored miles_pupil must reproduce the training pupil.

    Golden values measured from the original training-sim module on the
    training grid (256 px over 8.1795 m, outer = 7.79/7.92). Any change
    to the vendored file or an hcipy behavior shift shows up here.
    """
    hcipy = pytest.importorskip("hcipy")
    from fpwfsc.tokyo_drift import miles_pupil

    grid = hcipy.make_pupil_grid(256, 8.1795)
    pupil = np.asarray(
        miles_pupil.generate_pupil(outer=7.79 / 7.92, pupil_grid=grid))
    assert pupil.size == 256 * 256
    assert pupil.min() >= 0.0 and pupil.max() <= 1.0
    assert pupil.sum() == pytest.approx(39755.09375, rel=1e-6)
