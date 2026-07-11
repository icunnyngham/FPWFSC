"""Mode-registry and vendored-pupil tests."""
import numpy as np
import pytest

from fpwfsc.tokyo_drift import mode_registry as reg


def test_registered_modes():
    modes = reg.list_modes()
    assert {"vampires_f760_10zern", "vampires_f750_35zern",
            "vampires_vvc_f750_35zern_crop"} <= set(modes)


@pytest.mark.parametrize("mode,n_modes,filt", [
    ("vampires_f760_10zern", 10, "F760"),
    ("vampires_f750_35zern", 35, "F750"),
    ("vampires_vvc_f750_35zern_crop", 35, "F750"),
])
def test_mode_config_and_manifest(mode, n_modes, filt):
    assert reg.ts2_config_path(mode).is_file()
    assert reg.mode_n_modes(mode) == n_modes
    manifest = reg.load_manifest(mode)
    assert manifest["instrument"] == "Vampires"
    assert manifest["filter"] == filt
    assert manifest["checkpoint"]  # pointer set (file itself is gitignored)


def test_mode_zernike_diameter():
    """The TranslationDM must command modes on the same basis diameter the
    sim renders on. No-coro modes use 7.79; the coro modes use the runtime
    auto-derived value (~7.9), so this must not be hardcoded."""
    assert reg.mode_zernike_diameter("vampires_f760_10zern") == pytest.approx(7.79)
    assert reg.mode_zernike_diameter(
        "vampires_vvc_f750_35zern_crop") == pytest.approx(7.9053295407)


def test_checkpoint_resolves_under_checkpoints_dir(tmp_path):
    """A bare manifest filename resolves to checkpoints/<mode>/<file>."""
    modes = tmp_path / "modes"
    (modes / "m").mkdir(parents=True)
    (modes / "m" / "manifest.yaml").write_text(
        "mode: m\ncheckpoint: weights.pt\n")
    ckpts = tmp_path / "checkpoints"
    (ckpts / "m").mkdir(parents=True)
    (ckpts / "m" / "weights.pt").write_bytes(b"stub")
    resolved = reg.checkpoint_path("m", modes_dir=modes, checkpoints_dir=ckpts)
    assert resolved == ckpts / "m" / "weights.pt"


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


def test_vendored_synthpsf_matches_training_generator():
    """The vendored miles_synthpsf (the coro-era pupil generator, distinct
    from miles_pupil — it has the spider_scale arg the VVC Lyot needs) must
    reproduce the training aperture and Lyot on the training grid."""
    hcipy = pytest.importorskip("hcipy")
    from fpwfsc.tokyo_drift import miles_synthpsf as ms

    grid = hcipy.make_pupil_grid(256, 8.1795)
    aperture = np.asarray(
        ms.generate_pupil(outer=0.9835858585858586, pupil_grid=grid))
    lyot = np.asarray(ms.generate_pupil(
        outer=0.9, inner=0.43, scale=1.4, spider_scale=1.6, pupil_grid=grid))
    assert aperture.sum() == pytest.approx(39628.26562, rel=1e-6)
    assert lyot.sum() == pytest.approx(26961.79688, rel=1e-6)
