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
        self.last_frames = None  # set by take_image, like the real class
        rng = np.random.default_rng(7)
        # A static PSF-ish frame: bright blob at the frame center.
        y, x = np.mgrid[0:frame_size, 0:frame_size]
        c = frame_size / 2
        self.frame = (1000.0 * np.exp(-((x - c) ** 2 + (y - c) ** 2) / 18.0)
                      + rng.normal(10.0, 1.0, (frame_size, frame_size)))

    def take_image(self, average=1):
        self.requested_averages.append(average)
        # Mirror bench_hardware.Vampires: stash the individual native-
        # dtype readouts of this grab as (N, y, x).
        self.last_frames = np.repeat(
            np.clip(self.frame, 0, None)[np.newaxis], average,
            axis=0).astype(np.uint16)
        return self.frame.copy()


class FakeSCEXAO:
    """Duck-typed stand-in for bench_hardware.SCEXAO.

    Instances register themselves so tests can reach the injection
    wrapper run() auto-builds via ``type(aosystem)(dm_channel=...)``.
    """

    instances = []

    def __init__(self, dm_channel="dm00disp04"):
        self.dm_channel = dm_channel
        self.commands = []
        FakeSCEXAO.instances.append(self)

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


def test_hardware_branch_logs_camera_frames_and_provenance(
        hardware_profile, tmp_path):
    """[IO] 'save camera frames' writes each iteration's pre-reduction
    readout cube in its native dtype, and every logged session carries a
    copy of the profile plus the subtracted background."""
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cam, ao = FakeVampires(), FakeSCEXAO()
    cfg = _hw_config(hardware_profile,
                     **{"IO.save_log": "True",
                        "IO.log_path": str(tmp_path),
                        "IO.save camera frames": "True"})
    run(cam, ao, config=cfg, configspec=SPEC)

    fits = pytest.importorskip("astropy.io.fits")
    session, = tmp_path.glob("tokyo_drift_*")
    assert (session / "calibration_profile.yaml").read_text() == \
        Path(hardware_profile).read_text()
    background = fits.getdata(session / "background.fits")
    np.testing.assert_array_equal(background, cam.dark)
    for iter_dir in sorted(session.glob("iter_*")):
        cube = fits.getdata(iter_dir / "camera_raw.fits")
        assert cube.shape == (2, 256, 256)  # SNR 'frames to average' = 2
        assert cube.dtype == np.uint16      # native dtype, pre-reduction


def test_hardware_repeats_zero_the_dm_between_episodes(hardware_profile):
    """Each new episode must reset the DM through the command path: the
    loop images before it commands, so without the zeroing the next
    episode's first frame would see the previous converged command."""
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cam, ao = FakeVampires(), FakeSCEXAO()
    cfg = _hw_config(hardware_profile,
                     **{"SIMULATION.n repeats": "2"})
    result = run(cam, ao, config=cfg, configspec=SPEC)

    assert len(result["repeats"]) == 2
    # random_walk, 2 iters/episode, no initial move: commands are
    # [ep1 iter1, ep1 iter2, ZERO, ep2 iter1, ep2 iter2]
    assert len(ao.commands) == 5
    np.testing.assert_array_equal(ao.commands[2], np.zeros((50, 50)))
    assert np.any(ao.commands[1] != 0) and np.any(ao.commands[3] != 0)


def test_hardware_safety_abort_zeros_the_dm(hardware_profile):
    """On a safety abort the refused command is never sent and the DM
    is left zeroed - not parked on the last (near-limit) command."""
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cam, ao = FakeVampires(), FakeSCEXAO()
    cfg = _hw_config(hardware_profile,
                     **{"DM.max actuator stroke (um)": "1e-9"})
    result = run(cam, ao, config=cfg, configspec=SPEC)

    assert result["loop"]["aborted"] == "dm_safety"
    assert result["loop"]["iterations"] == 0
    # The only command ever sent is the post-abort zero
    assert len(ao.commands) == 1
    np.testing.assert_array_equal(ao.commands[0], np.zeros((50, 50)))


def test_hardware_wfe_injection_records_and_zeros(hardware_profile,
                                                  tmp_path):
    """[SIMULATION] 'injection dm channel': per episode the drawn WFE
    goes out on the second channel and into the log; when the run ends
    BOTH channels are zeroed."""
    import json
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    FakeSCEXAO.instances.clear()
    cam, ao = FakeVampires(), FakeSCEXAO()
    cfg = _hw_config(hardware_profile,
                     **{"SIMULATION.n repeats": "2",
                        "SIMULATION.injection dm channel": "dm00disp06",
                        "SIMULATION.initial error rms": "0.05",
                        "SIMULATION.wfe seed": "3",
                        "IO.save_log": "True",
                        "IO.log_path": str(tmp_path)})
    result = run(cam, ao, config=cfg, configspec=SPEC)

    # run() auto-built the injector as type(aosystem)(dm_channel=...)
    injector, = [i for i in FakeSCEXAO.instances
                 if i.dm_channel == "dm00disp06"]
    # One injection per episode, then the end-of-run zero
    assert len(injector.commands) == 3
    assert np.any(injector.commands[0] != 0)
    assert np.any(injector.commands[1] != 0)
    assert np.any(injector.commands[0] != injector.commands[1])
    np.testing.assert_array_equal(injector.commands[2], np.zeros((50, 50)))
    # Correction channel also ends zeroed (injection-specific solution)
    np.testing.assert_array_equal(ao.commands[-1], np.zeros((50, 50)))

    # Coefficients are recorded per episode - no longer null on hardware
    assert len(result["repeats"]) == 2
    fits = pytest.importorskip("astropy.io.fits")
    sessions = sorted(tmp_path.glob("tokyo_drift_*"))
    assert len(sessions) == 2
    for session, injected, rep in zip(sessions, injector.commands, result["repeats"]):
        with open(session / "episode.json") as f:
            episode = json.load(f)
        assert episode["injected_error_coeffs"] is not None
        np.testing.assert_allclose(rep["injected_error_coeffs"],
                                   episode["injected_error_coeffs"])
        np.testing.assert_allclose(
            fits.getdata(session / "injected_command.fits"), injected)


def test_injection_channel_must_differ_from_correction(hardware_profile):
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    cfg = _hw_config(hardware_profile,
                     **{"SIMULATION.injection dm channel": "dm00disp04"})
    with pytest.raises(ValueError, match="must differ"):
        run(FakeVampires(), FakeSCEXAO(), config=cfg, configspec=SPEC)


def test_refused_injection_aborts_episode_and_zeros(hardware_profile):
    """An injection draw over the DM limits is never sent: the episode
    aborts like any safety trip, and the channels end zeroed."""
    pytest.importorskip("telescope_sim")
    from fpwfsc.tokyo_drift.run import run
    FakeSCEXAO.instances.clear()
    cam, ao = FakeVampires(), FakeSCEXAO()
    cfg = _hw_config(hardware_profile,
                     **{"SIMULATION.injection dm channel": "dm00disp06",
                        "SIMULATION.initial error rms": "50.0"})
    result = run(cam, ao, config=cfg, configspec=SPEC)

    loop = result["loop"]
    assert loop["aborted"] == "dm_safety"
    assert "injected WFE command" in loop["safety_error"]
    assert loop["iterations"] == 0
    injector, = [i for i in FakeSCEXAO.instances
                 if i.dm_channel == "dm00disp06"]
    # Refused injection never sent; only the end-of-run zero went out
    assert len(injector.commands) == 1
    np.testing.assert_array_equal(injector.commands[0], np.zeros((50, 50)))


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
