"""run.py real-hardware branch: exercised with fake wrapper objects.

These tests guard the exact code path a bench deployment hits — no
NotImplementedError, filter assertion, dark wiring into the reducer,
raw 50x50 float32 commands to set_dm_data, and the averaging
pass-through — without any SCExAO packages present.
"""
from pathlib import Path

import numpy as np
import pytest
from configobj import ConfigObj

pytest.importorskip("hcipy")

PIPELINE_DIR = Path(__file__).resolve().parents[1]
SPEC = str(PIPELINE_DIR / "tokyo_drift_config.spec")
SIM_INI = str(PIPELINE_DIR / "tokyo_drift_config_sim.ini")

MODE = "vampires_f760_10zern"  # F760, 10 modes, 128x128 focal plane


class FakeVampires:
    """Duck-typed stand-in for bench_hardware.Vampires."""

    def __init__(self, filter_name="760-50", frame_size=256):
        self.filter_name = filter_name
        self.dark = np.zeros((frame_size, frame_size))
        self.dark_info = "fake dark"
        self.requested_averages = []
        rng = np.random.default_rng(7)
        # A static PSF-ish frame: bright blob at the frame center.
        y, x = np.mgrid[0:frame_size, 0:frame_size]
        c = frame_size / 2
        self.frame = (1000.0 * np.exp(-((x - c) ** 2 + (y - c) ** 2) / 18.0)
                      + rng.normal(10.0, 1.0, (frame_size, frame_size)))

    def take_image(self, average=1):
        self.requested_averages.append(average)
        return self.frame.copy()


class FakeSCEXAO:
    """Duck-typed stand-in for bench_hardware.SCEXAO."""

    def __init__(self):
        self.commands = []

    def set_dm_data(self, command):
        self.commands.append(np.asarray(command))


@pytest.fixture()
def hardware_profile(tmp_path):
    yaml = pytest.importorskip("yaml")
    profile = {
        "mode": MODE,
        "dm_scale": 1.5,
        "dm_rot_deg": 0.0,
        "dm_flip_x": False,
        "dm_flip_y": False,
        "image_rot_deg": 0.0,
        "crop_cx": 128,
        "crop_cy": 128,
        "flip_x": False,
        "flip_y": False,
        "shift_x": 1,   # bench-only shifts: allowed on hardware
        "shift_y": 2,
    }
    path = tmp_path / "hw_profile.yaml"
    path.write_text(yaml.safe_dump(profile))
    return str(path)


def _hw_config(profile, **overrides):
    cfg = ConfigObj(SIM_INI)
    cfg["MODE"]["mode name"] = MODE
    cfg["MODE"]["calibration profile"] = profile
    cfg["LOOP_SETTINGS"]["N iter"] = "2"
    cfg["LOOP_SETTINGS"]["predictor"] = "random_walk"
    cfg["LOOP_SETTINGS"]["strehl early stop"] = "None"
    cfg["SNR"]["frames to average"] = "2"
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        cfg[section][key] = value
    return cfg


def test_hardware_branch_runs_the_loop(hardware_profile):
    pytest.importorskip("telescope_sim")  # ideal reference PSF
    from fpwfsc.tokyo_drift.run import run
    cam, ao = FakeVampires(), FakeSCEXAO()
    result = run(cam, ao, config=_hw_config(hardware_profile),
                 configspec=SPEC)

    assert result["loop"]["iterations"] == 2
    assert result["truth"] is None
    assert result["injected_error_coeffs"] is None
    # Raw 50x50 command contract straight to set_dm_data
    assert len(ao.commands) >= 2
    for cmd in ao.commands:
        assert cmd.shape == (50, 50)
        assert cmd.dtype == np.float32
    # The configured frame averaging reaches the camera untouched
    assert set(cam.requested_averages) == {2}


def test_hardware_branch_refuses_wrong_filter(hardware_profile):
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cam = FakeVampires(filter_name="675-50")  # mode is trained on F760
    with pytest.raises(ValueError, match="does not match mode"):
        run(cam, FakeSCEXAO(), config=_hw_config(hardware_profile),
            configspec=SPEC)


def test_hardware_branch_refuses_oracle(hardware_profile):
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cfg = _hw_config(hardware_profile,
                     **{"LOOP_SETTINGS.predictor": "oracle"})
    with pytest.raises(ValueError, match="oracle"):
        run(FakeVampires(), FakeSCEXAO(), config=cfg, configspec=SPEC)


def test_assert_camera_matches_mode_warns_without_filter_name():
    from fpwfsc.tokyo_drift.run import assert_camera_matches_mode

    class NoFilterCam:
        pass

    with pytest.warns(UserWarning, match="no filter_name"):
        assert_camera_matches_mode(NoFilterCam(), MODE)


def test_calibration_runs_against_hardware_bench():
    """The GUI's Calibrate path on real hardware: a HardwareBench (no
    .truth) must run the full fit and return a profile plus a report
    WITHOUT the sim-only recovery keys. The fitted values are garbage
    against a static fake frame — only completion and shape matter."""
    pytest.importorskip("telescope_sim")  # ideal probe references
    from fpwfsc.tokyo_drift.calibration.harness import (
        HardwareBench, calibrate_bench_sim)

    cam, ao = FakeVampires(), FakeSCEXAO()
    dark = cam.dark
    bench = HardwareBench(cam, ao, reduce=lambda f: f - dark)
    profile, report = calibrate_bench_sim(MODE, bench=bench, average=1)

    for key in ("image_rot_deg", "crop_cx", "crop_cy",
                "flip_x", "flip_y", "dm_scale"):
        assert key in profile
    assert "truth" not in report
    assert "image_rot_error_deg" not in report
    # The probe pokes went out as raw 50x50 float32 commands
    assert ao.commands and all(c.shape == (50, 50) for c in ao.commands)
    assert cam.requested_averages  # frames actually came from the camera
