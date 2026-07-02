"""Calibration tests: fitters on synthetic scenes, profile persistence,
and the M6a recovery checkpoint (fit against the bench sim without
truth access, compare against the injected truth)."""
import numpy as np
import pytest
from scipy.ndimage import rotate as nd_rotate

from fpwfsc.tokyo_drift.calibration.manual import (
    fit_center,
    fit_flips,
    fit_rotation,
    probe_coefficients,
)
from fpwfsc.tokyo_drift.calibration.profiles import (
    assert_sim_safe,
    load_profile,
    save_profile,
)


def _asym_pattern(size, row, col):
    """Primary blob + weaker companion along +x + faint one at +y:
    breaks both centro- and mirror-symmetry."""
    yy, xx = np.mgrid[:size, :size]

    def blob(r, c, s=2.5):
        return np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * s ** 2))

    return blob(row, col) + 0.5 * blob(row, col + 14) + 0.25 * blob(row + 9, col)


# --- Fitters on synthetic scenes ----------------------------------------

@pytest.fixture()
def synthetic():
    ref = _asym_pattern(64, 32, 32)
    scene = _asym_pattern(300, 158, 144)          # off-center in a big field
    raw = nd_rotate(scene, -33.0, reshape=False, order=1)  # camera rot
    return raw, ref


def test_fit_rotation_recovers_synthetic_angle(synthetic):
    raw, ref = synthetic
    result = fit_rotation(raw, ref, crop_res=64, angle_range=(0, 90),
                          coarse_step=2.0)
    # ~0.6 deg bias is inherent to this sparse 3-blob toy scene
    # (interpolation + crop quantization); the sharp accuracy pin is
    # the bench-sim recovery test (<0.5 deg on a real PSF).
    assert abs(result["image_rot_deg"] - 33.0) <= 1.0
    assert result["preview"].shape == (64, 64)


def test_fit_center_locates_pattern(synthetic):
    raw, ref = synthetic
    result = fit_center(raw, ref, 33.0, crop_res=64)
    crop = result["preview"]
    # The primary blob sits at the crop center (+-1 px conv parity)
    peak = np.unravel_index(crop.argmax(), crop.shape)
    assert abs(peak[0] - 32) <= 2 and abs(peak[1] - 32) <= 2


def test_fit_flips_detects_mirrored_scene():
    ref = _asym_pattern(64, 32, 32)
    scene = np.fliplr(_asym_pattern(300, 150, 150))
    center = fit_center(scene, ref, 0.0, crop_res=64)
    result = fit_flips(scene, ref, 0.0, center["crop_cx"],
                       center["crop_cy"], crop_res=64)
    assert result["flip_x"] is True
    assert result["flip_y"] is False


def test_probe_uses_odd_modes():
    probe = probe_coefficients(10)
    assert probe[5] != 0 and probe[8] != 0  # Noll 7 + Noll 10
    assert np.count_nonzero(probe) == 2


# --- Profile persistence -------------------------------------------------

def test_profile_save_load_round_trip(tmp_path):
    pytest.importorskip("yaml")
    profile = {"mode": "vampires_f760_10zern", "image_rot_deg": 235.4,
               "crop_cx": 270, "crop_cy": 268, "dm_scale": 1.3}
    path = save_profile("bench_test", profile, calibrations_dir=tmp_path)
    assert path.name == "bench_test.yaml"

    by_name = load_profile("bench_test", calibrations_dir=tmp_path)
    by_path = load_profile(str(path))
    assert by_name == by_path
    assert by_name["image_rot_deg"] == 235.4
    # Defaults filled for unset fields
    assert by_name["shift_x"] == 0 and by_name["dm_rot_deg"] == 0.0
    assert by_name["flip_x"] is False
    assert "saved_at" in by_name


def test_load_unknown_profile_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profile("nope", calibrations_dir=tmp_path)


def test_sim_guard_refuses_shifts():
    assert_sim_safe({"shift_x": 0, "shift_y": 0})  # fine
    with pytest.raises(ValueError, match="shift"):
        assert_sim_safe({"shift_x": 1, "shift_y": 2})


# --- Recovery checkpoint (bench sim, no truth access) --------------------

@pytest.fixture(scope="module")
def easy_recovery():
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.calibration.harness import calibrate_bench_sim
    stages = []
    profile, report = calibrate_bench_sim(
        "vampires_f760_10zern", preset="easy", seed=27,
        stage_callback=lambda payload: stages.append(payload))
    return profile, report, stages


def test_recovery_on_easy_preset(easy_recovery):
    profile, report, _stages = easy_recovery
    assert abs(report["image_rot_error_deg"]) < 0.5
    assert abs(report["dm_scale_error_frac"]) < 0.06  # scale-grid quantum
    assert report["flips_expected_false"]


def test_stage_callback_streams_fit_progress(easy_recovery):
    _profile, _report, stages = easy_recovery
    assert [s["stage"] for s in stages] == [
        "probe", "rotation", "center", "flips", "scale"]
    for stage in stages:
        assert stage["preview"] is not None
        assert stage["reference"] is not None
    # Sweep curves ride along for the GUI's score plot
    assert stages[1]["curve"] is not None      # rotation sweep
    assert stages[4]["curve"] is not None      # scale grid
    # Params accumulate across stages
    assert "image_rot_deg" in stages[1]["params"]
    assert "dm_scale" in stages[4]["params"]


def test_recovery_on_realistic_preset():
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.calibration.harness import calibrate_bench_sim
    profile, report = calibrate_bench_sim(
        "vampires_f760_10zern", preset="realistic_vampires", seed=5)
    assert abs(report["image_rot_error_deg"]) < 1.0
    assert abs(report["dm_scale_error_frac"]) < 0.06
    assert report["flips_expected_false"]


def test_recovery_on_preset_1_documented_degradation():
    """preset_1 couples a large DM rotation (-6.25 deg) with a 1.3x
    scale error; the v1 single-probe fit degrades there (image-rot fit
    partially absorbs the DM rotation, biasing the scale match). Pin
    the current degradation envelope so improvements/regressions show;
    the v2 joint global fit is the real answer."""
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.calibration.harness import calibrate_bench_sim
    profile, report = calibrate_bench_sim("vampires_f760_10zern",
                                          preset="preset_1", seed=1)
    assert abs(report["image_rot_error_deg"]) < 6.0
    assert abs(report["dm_scale_error_frac"]) < 0.35


def test_fitted_profile_drives_converging_loop(easy_recovery, tmp_path):
    """End-to-end M6a checkpoint: calibrate (no truth access), save the
    profile, run the oracle loop with it, converge."""
    pytest.importorskip("telescope_sim")
    from pathlib import Path

    from configobj import ConfigObj
    from fpwfsc.tokyo_drift.run import run

    pipeline_dir = Path(__file__).resolve().parents[1]
    SIM_INI = str(pipeline_dir / "tokyo_drift_config_sim.ini")
    SPEC = str(pipeline_dir / "tokyo_drift_config.spec")

    profile, _report, _stages = easy_recovery
    path = save_profile("fitted_easy_27", profile, calibrations_dir=tmp_path)

    cfg = ConfigObj(SIM_INI)
    cfg["MODE"]["calibration profile"] = str(path)
    cfg["LOOP_SETTINGS"]["N iter"] = 12
    cfg["LOOP_SETTINGS"]["strehl early stop"] = 0.9
    result = run("Sim", "Sim", config=cfg, configspec=SPEC)
    strehls = result["loop"]["strehls"]
    assert np.nanmax(strehls) >= 0.9
