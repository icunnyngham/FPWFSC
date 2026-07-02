"""GUI-layer tests for tokyo_drift.

Qt widgets are exercised with the offscreen platform plugin so the
suite runs headless. The pure logic (registry listing, display
normalization) is tested without Qt.
"""
import os

# Must be set before any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from fpwfsc.tokyo_drift import gui_helper as helper

PIPELINE_DIR = Path(__file__).resolve().parents[1]
SPEC = str(PIPELINE_DIR / "tokyo_drift_config.spec")
SIM_INI = str(PIPELINE_DIR / "tokyo_drift_config_sim.ini")


# --- Registry listing helpers (no Qt) ---------------------------------

def test_list_modes_missing_dir_is_empty(tmp_path):
    assert helper.list_modes(modes_dir=tmp_path / "nope") == []


def test_list_modes_finds_subdirs(tmp_path):
    (tmp_path / "vampires_f760_10zern").mkdir()
    (tmp_path / "vampires_f750_35zern").mkdir()
    (tmp_path / "_hidden").mkdir()
    (tmp_path / "stray_file.yaml").touch()
    assert helper.list_modes(modes_dir=tmp_path) == [
        "vampires_f750_35zern", "vampires_f760_10zern"]


def test_list_calibrations_filters_by_mode(tmp_path):
    yaml = pytest.importorskip("yaml")
    (tmp_path / "bench_jul01.yaml").write_text(
        yaml.safe_dump({"mode": "vampires_f760_10zern", "image_rot_deg": 235.4}))
    (tmp_path / "other_mode.yaml").write_text(
        yaml.safe_dump({"mode": "vampires_f750_35zern"}))
    assert helper.list_calibrations(
        mode_name="vampires_f760_10zern",
        calibrations_dir=tmp_path) == ["bench_jul01"]
    assert helper.list_calibrations(calibrations_dir=tmp_path) == [
        "bench_jul01", "other_mode"]


# --- Plotter display math (no Qt event loop needed) -------------------

def test_log_display_normalizes_to_unit_range():
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import log_display
    image = np.abs(np.random.default_rng(0).normal(size=(32, 32))) + 1e-3
    disp = log_display(image)
    assert disp.min() == pytest.approx(0.0)
    assert disp.max() == pytest.approx(1.0)
    # Log scaling is monotonic: brightest pixel stays brightest.
    assert np.unravel_index(disp.argmax(), disp.shape) == \
        np.unravel_index(image.argmax(), image.shape)


def test_log_display_handles_empty_image():
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import log_display
    assert not np.any(log_display(np.zeros((8, 8))))


def test_inferno_lut_shape():
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import inferno_lut
    lut = inferno_lut()
    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    # inferno runs dark -> bright
    assert lut[0].sum() < lut[-1].sum()


def test_gui_module_imports_in_script_mode():
    """`python fpwfsc/tokyo_drift/tokyo_drift_GUI.py` is the documented
    way to launch FPWFSC GUIs — emulate that import context (no parent
    package, script dir on sys.path) and make sure the module loads."""
    import importlib.util
    import sys

    gui_path = PIPELINE_DIR / "tokyo_drift_GUI.py"
    sys.path.insert(0, str(PIPELINE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "td_gui_scriptmode", gui_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.pf.__name__ == "tokyo_drift_plotter_qt"
        assert hasattr(mod, "TokyoDriftConfigGUI")
    finally:
        sys.path.remove(str(PIPELINE_DIR))


# --- Offscreen Qt tests -------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_plotter_updates_offscreen(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import LivePlotter

    plotter = LivePlotter()
    try:
        yy, xx = np.mgrid[-16:16, -16:16]
        ideal = np.exp(-(xx**2 + yy**2) / (2 * 3.0**2))
        source = np.roll(ideal, 2, axis=0)
        strehls = np.full(10, np.nan)
        strehls[0] = 0.5
        # Same-thread emit -> direct connection -> synchronous update.
        plotter.update({"n_iter": 10, "iteration": 1, "source": source,
                        "ideal": ideal, "strehls": strehls,
                        "mode_coeffs": np.zeros(10)})
        assert plotter._raw_images["source"] is not None
        assert plotter._raw_images["ideal"] is not None
        assert plotter._raw_images["diff"] is not None
        assert plotter._raw_images["diff"].shape == ideal.shape
    finally:
        plotter.close()


class _StubPlotter:
    closed = False

    def __init__(self):
        self.payloads = []

    def update(self, payload):
        self.payloads.append(payload)


def _fake_calibration(gui):
    """Feed the GUI a completed calibration without running one."""
    yy, xx = np.mgrid[-64:64, -64:64]
    raw = np.exp(-((xx - 5) ** 2 + (yy + 3) ** 2) / (2 * 4.0 ** 2))
    ref = np.exp(-(xx[32:96, 32:96] ** 2 + yy[32:96, 32:96] ** 2)
                 / (2 * 4.0 ** 2))
    profile = {"image_rot_deg": 5.0, "crop_cx": 64, "crop_cy": 64,
               "flip_x": False, "flip_y": True, "dm_scale": 1.2,
               "dm_rot_deg": 0.0}
    report = {"stage_previews": {"raw": raw}, "reference_psf": ref,
              "probe_coefficients": np.zeros(10), "corrector": "zernike_dm",
              "image_rot_error_deg": 0.0, "dm_scale_error_frac": 0.0,
              "flips_expected_false": True}
    gui.on_calibration_done(profile, report, None, None)
    return profile


def test_calibration_workbench_fields_and_live_preview(qapp, tmp_path):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        profile = _fake_calibration(gui)
        # Fields populated from the fitted profile
        assert gui.calib_fields["image_rot_deg"].text() == "5.0"
        assert gui.calib_fields["flip_y"].currentText() == "True"
        assert gui._profile_from_fields()["dm_scale"] == 1.2

        # Hand-edit + live re-render through a stub plotter
        gui.calib_plotter = _StubPlotter()
        gui.calib_fields["image_rot_deg"].setText("7.5")
        gui.calib_fields["crop_cx"].setText("None")  # -> auto center
        gui.refresh_calibration_preview()
        parsed = gui._profile_from_fields()
        assert parsed["image_rot_deg"] == 7.5
        assert parsed["crop_cx"] is None
        assert len(gui.calib_plotter.payloads) == 1
        payload = gui.calib_plotter.payloads[0]
        assert payload["source"].shape == (64, 64)
        assert payload["source_title"] == "Calibration: manual edit"

        # Save through a patched name dialog into a scratch registry
        from PyQt5.QtWidgets import QInputDialog
        original = QInputDialog.getText
        QInputDialog.getText = staticmethod(
            lambda *a, **k: ("workbench_test", True))
        try:
            gui.calibrations_dir = tmp_path
            gui.on_save_calibration()
        finally:
            QInputDialog.getText = original
        assert (tmp_path / "workbench_test.yaml").is_file()
        names = [gui.calibration_select.itemText(i)
                 for i in range(gui.calibration_select.count())]
        assert "workbench_test" in names
        assert gui.config['MODE']['calibration profile'] == "workbench_test"

        # Selecting the saved profile loads it back into the fields
        gui.calib_fields["image_rot_deg"].setText("0.0")
        gui.on_calibration_selected("workbench_test")
        assert gui._profile_from_fields()["image_rot_deg"] == 7.5
        assert profile["flip_y"] is True  # untouched original
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_plotter_renders_curve_payload(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import LivePlotter

    plotter = LivePlotter()
    try:
        plotter.update({"curve": ([0, 1, 2], [0.1, 0.9, 0.3]),
                        "curve_title": "Rotation sweep score"})
        x, y = plotter.strehl_curve.getData()
        assert list(x) == [0, 1, 2]
        assert y[1] == pytest.approx(0.9)
    finally:
        plotter.close()


def test_gui_constructs_and_round_trips(qapp, tmp_path):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI
    from fpwfsc.tokyo_drift.run import run

    gui = TokyoDriftConfigGUI()
    try:
        # Selectors populated
        hardware = [gui.hardware_select.itemText(i)
                    for i in range(gui.hardware_select.count())]
        assert hardware == helper.valid_instruments
        # The configured mode is selectable even before modes/ exists
        assert gui.mode_select.currentText() == "vampires_f760_10zern"
        calibrations = [gui.calibration_select.itemText(i)
                        for i in range(gui.calibration_select.count())]
        assert "None" in calibrations
        # Sim hardware resolved to the sentinels
        assert (gui.camera, gui.aosystem) == ("Sim", "Sim")

        # GUI -> config -> .ini -> headless run() round trip (stop event
        # set: this validates the config contract, not the loop)
        gui.update_config_from_gui()
        saved = tmp_path / "gui_roundtrip.ini"
        gui.config.filename = str(saved)
        gui.config.write()

        import threading
        stop = threading.Event()
        stop.set()
        settings = run("Sim", "Sim", config=str(saved), configspec=SPEC,
                       my_event=stop)["settings"]
        assert settings["MODE"]["mode name"] == "vampires_f760_10zern"
        assert settings["MODE"]["calibration profile"] is None
        assert settings["LOOP_SETTINGS"]["Plot"] is True
    finally:
        gui.thread_check_timer.stop()
        gui.close()
