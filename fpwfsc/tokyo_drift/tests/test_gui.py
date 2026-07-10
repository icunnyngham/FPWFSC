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

def test_log_display_gives_decades_below_peak():
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import (
        LOG_DECADES, log_display)
    image = np.abs(np.random.default_rng(0).normal(size=(32, 32))) + 1e-3
    disp = log_display(image)
    # Physical units: 0 at the peak, clipped LOG_DECADES below it —
    # a colorbar over these values reads directly as log10 contrast.
    assert disp.max() == pytest.approx(0.0, abs=1e-6)
    assert disp.min() >= -LOG_DECADES
    # Log scaling is monotonic: brightest pixel stays brightest.
    assert np.unravel_index(disp.argmax(), disp.shape) == \
        np.unravel_index(image.argmax(), image.shape)


def test_log_display_handles_empty_image():
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import (
        LOG_DECADES, log_display)
    assert np.all(log_display(np.zeros((8, 8))) == -LOG_DECADES)


def test_bench_preset_field_is_a_dropdown():
    from fpwfsc.tokyo_drift import gui_helper as gh
    choices = gh.get_choices("ALIGNMENT", "bench sim preset")
    assert set(choices) >= {"easy", "realistic_vampires", "stress_test",
                            "vampires_2024_measured"}
    assert gh.get_choices("SIMULATION", "seed") is None


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


def test_no_dotted_relative_imports_in_gui():
    """Script-mode launch has no parent package, so a dotted relative
    import (`from .mode_registry import ...`) fails at runtime — and the
    module-load test above can't catch ones buried in methods (they only
    execute on a user action). Every intra-package import must be absolute
    `fpwfsc.tokyo_drift...`; the one same-dir plotter uses the bare
    `from . import tokyo_drift_plotter_qt` form with a try/except
    fallback, which this pattern deliberately does not flag."""
    import re
    src = (PIPELINE_DIR / "tokyo_drift_GUI.py").read_text()
    offenders = re.findall(r'^\s*from \.\w[\w.]* import.*$', src,
                           re.MULTILINE)
    assert offenders == [], (
        "dotted relative imports break script-mode launch; make these "
        f"absolute (fpwfsc.tokyo_drift...): {offenders}")


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
              "dm_scale_over_truth": 1.5, "flips_expected_false": True}
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
        assert payload["source_title"].startswith("Calibration: manual edit")
        assert "RMS diff" in payload["source_title"]

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


def test_plotter_renders_curve_payload_on_coeff_panel(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import LivePlotter

    plotter = LivePlotter()
    try:
        plotter.update({"curve": ([0, 1, 2], [0.1, 0.9, 0.3]),
                        "curve_title": "Rotation sweep score"})
        x, y = plotter.sweep_curve.getData()
        assert list(x) == [0, 1, 2]
        assert y[1] == pytest.approx(0.9)
    finally:
        plotter.close()


def test_plotter_renders_progress_with_stage_marks(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import LivePlotter

    plotter = LivePlotter()
    try:
        plotter.update({"progress": {
            "x": [0, 1, 2], "y": [0.3, 0.1, 0.05],
            "stage_marks": [(0, "rotation"), (1, "center"), (2, "flips")],
            "ylabel": "RMS(diff)"}})
        x, y = plotter.strehl_curve.getData()
        assert list(x) == [0, 1, 2]
        # One dashed line + one text label per stage
        assert len(plotter._stage_marks) == 6
        # A loop strehls payload reclaims the axes and clears the marks
        plotter.update({"n_iter": 5, "strehls": np.array([0.5, np.nan,
                                                          np.nan, np.nan,
                                                          np.nan])})
        assert plotter._stage_marks == []
    finally:
        plotter.close()


def test_plotter_diff_colorbar_tracks_residual_scale(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_plotter_qt import LivePlotter

    plotter = LivePlotter()
    try:
        yy, xx = np.mgrid[-16:16, -16:16]
        ideal = np.exp(-(xx**2 + yy**2) / (2 * 3.0**2))
        source = np.roll(ideal, 2, axis=0)
        plotter.update({"source": source, "ideal": ideal})
        lo, hi = plotter.color_bars["diff"].levels()
        span = float(np.max(np.abs(plotter._raw_images["diff"])))
        assert hi == pytest.approx(span)
        assert lo == pytest.approx(-span)
    finally:
        plotter.close()


def test_gui_stage_handler_builds_progress(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        gui.calib_plotter = _StubPlotter()
        ref = np.exp(-(np.mgrid[-32:32, -32:32][0]**2
                       + np.mgrid[-32:32, -32:32][1]**2) / 18.0)
        # Raw probe stage: preview shape mismatches the reference ->
        # no progress point yet
        gui.on_calibration_stage({"stage": "probe", "params": {},
                                  "preview": np.zeros((128, 128)),
                                  "reference": ref, "curve": None})
        assert gui._calib_progress["x"] == []
        # Rotation stage: matched shapes -> a point + annotated mark
        gui.on_calibration_stage({
            "stage": "rotation", "params": {"image_rot_deg": 12.0},
            "preview": np.roll(ref, 3, axis=1), "reference": ref,
            "curve": ([0, 1], [0.1, 0.9]),
            "curve_title": "Rotation sweep score"})
        assert gui._calib_progress["x"] == [0]
        assert gui._calib_progress["stage_marks"] == [(0, "rotation")]
        payload = gui.calib_plotter.payloads[-1]
        assert payload["progress"]["y"][0] > 0
        assert "RMS diff" in payload["source_title"]
        assert gui.calib_fields["image_rot_deg"].text() == "12.0"
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_calibration_fields_start_at_defaults(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        assert gui.calib_fields["image_rot_deg"].text() == "0.0"
        assert gui.calib_fields["dm_scale"].text() == "1.0"
        assert gui.calib_fields["crop_cx"].text() == "None"
        assert gui.calib_fields["flip_x"].currentText() == "False"
        # And they parse back to the profile defaults
        parsed = gui._profile_from_fields()
        assert parsed["dm_scale"] == 1.0 and parsed["crop_cx"] is None
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_resolve_run_profile_paths(qapp, tmp_path):
    import yaml
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        # Named profile selected: passes straight through, no dialog
        gui._ask_unsaved_run = lambda: pytest.fail("dialog must not open")
        gui.calibration_select.addItem("saved_prof")
        gui.calibration_select.setCurrentText("saved_prof")
        assert gui._resolve_run_profile() == "saved_prof"

        # No profile + cancel -> None
        gui.calibration_select.setCurrentText("None")
        gui._ask_unsaved_run = lambda: 'cancel'
        assert gui._resolve_run_profile() is None

        # No profile + run-without-saving -> ephemeral yaml from fields
        gui.calib_fields["image_rot_deg"].setText("123.4")
        gui._ask_unsaved_run = lambda: 'run'
        path = gui._resolve_run_profile()
        assert path is not None and path.endswith(".yaml")
        with open(path) as f:
            ephemeral = yaml.safe_load(f)
        assert ephemeral["image_rot_deg"] == 123.4
        assert ephemeral["mode"] == gui.mode_select.currentText()

        # 'save' chosen but the save dialog cancelled -> None
        gui._ask_unsaved_run = lambda: 'save'
        gui.on_save_calibration = lambda: None  # user cancels naming
        assert gui._resolve_run_profile() is None
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_view_ready_populates_workbench(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        gui.calib_plotter = _StubPlotter()
        raw = np.random.default_rng(0).random((256, 256))
        ref = np.random.default_rng(1).random((128, 128))
        gui.on_view_ready({"bench": "bench_obj", "ideal": None,
                           "raw": raw, "reference": ref,
                           "probe": np.zeros(10),
                           "corrector": "zernike_dm", "n_modes": 10,
                           "crop_res": 128})
        assert gui.calib_raw is raw
        assert gui.calib_bench == "bench_obj"
        payload = gui.calib_plotter.payloads[-1]
        assert payload["source_title"].startswith("View: raw detector")
        # Manual adjustment now works against the cached frame
        gui.calib_fields["image_rot_deg"].setText("10.0")
        gui.refresh_calibration_preview()
        assert gui.calib_plotter.payloads[-1]["source"].shape == (128, 128)
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_bench_preset_widget_is_dropdown(qapp):
    from PyQt5.QtWidgets import QComboBox
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        widget = gui.create_input_widget("ALIGNMENT", "bench sim preset",
                                         "easy")
        assert isinstance(widget, QComboBox)
        items = [widget.itemText(i) for i in range(widget.count())]
        assert "stress_test" in items
        assert widget.currentText() == "easy"
    finally:
        gui.thread_check_timer.stop()
        gui.close()


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


# --- Section display names / predictor labels / alignment (no Qt) ------

def test_section_display_name_round_trip():
    assert helper.section_display_name("SIMULATION") == "Test WFE injection params"
    assert helper.section_key_from_label(
        "Test WFE injection params") == "SIMULATION"
    # Unmapped sections are identity in both directions.
    assert helper.section_display_name("DM") == "DM"
    assert helper.section_key_from_label("DM") == "DM"


def test_predictor_option_labels_map_model_first():
    labels = helper.get_option_labels("LOOP_SETTINGS", "predictor")
    assert labels[0] == ("model", "Tokyo Drift (NN)")
    values = [v for v, _ in labels]
    assert values == ["model", "oracle", "random_walk"]


def test_bench_preset_is_sim_only_flag():
    assert helper.is_sim_only("ALIGNMENT", "bench sim preset") is True
    assert helper.is_sim_only("ALIGNMENT", "probe amplitude") is False
    assert helper.is_sim_only("SIMULATION", "seed") is False


# --- GUI wiring for the reorg (Qt) -------------------------------------

def test_predictor_dropdown_shows_labels_stores_values(qapp):
    from PyQt5.QtWidgets import QComboBox
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        widget = gui.create_input_widget("LOOP_SETTINGS", "predictor", "model")
        assert isinstance(widget, QComboBox)
        shown = [widget.itemText(i) for i in range(widget.count())]
        assert shown == ["Tokyo Drift (NN)", "Oracle (debug)",
                         "Random walk (debug)"]
        # The stored value is the internal token, not the display label.
        assert widget.currentData() == "model"
        assert gui.get_widget_value(widget) == "model"
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_alignment_section_syncs_to_config(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        gui.preset_combo.setCurrentText("stress_test")
        gui.probe_amp_field.setText("0.5")
        gui.update_config_from_gui()
        assert gui.config["ALIGNMENT"]["bench sim preset"] == "stress_test"
        assert float(gui.config["ALIGNMENT"]["probe amplitude"]) == 0.5
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_bench_preset_hidden_off_sim(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        # Default hardware is Sim -> the sim-only preset row shows.
        assert not gui.preset_row.isHidden()
        # Force a non-Sim selection without triggering the (SCExAO-only)
        # hardware loader, then re-apply visibility.
        gui.hardware_select.blockSignals(True)
        gui.hardware_select.setCurrentText("Vampires")
        gui.hardware_select.blockSignals(False)
        gui._apply_sim_only_visibility()
        assert gui.preset_row.isHidden()
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_model_availability_message(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        assert gui._model_availability_message("") is None
        msg = gui._model_availability_message("definitely_not_a_mode")
        assert msg is not None and "definitely_not_a_mode" in msg
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_delete_button_tracks_selection(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        # 'None' selected at start -> trashcan disabled.
        assert gui.calibration_select.currentText() == "None"
        assert not gui.delete_calibration_button.isEnabled()
        # A named config -> enabled.
        gui.calibration_select.addItem("some_prof")
        gui.calibration_select.setCurrentText("some_prof")
        assert gui.delete_calibration_button.isEnabled()
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_delete_calibration_removes_after_confirm(qapp, tmp_path, monkeypatch):
    import yaml
    from PyQt5.QtWidgets import QMessageBox
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        gui.calibrations_dir = tmp_path
        prof = tmp_path / "to_delete.yaml"
        prof.write_text(yaml.safe_dump(
            {"mode": "vampires_f760_10zern", "image_rot_deg": 1.0}))
        gui.populate_calibration_selector(select="to_delete")
        assert gui.calibration_select.currentText() == "to_delete"

        # Decline -> nothing happens.
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.No)
        gui.on_delete_calibration()
        assert prof.exists()

        # Confirm -> file gone, selector drops to None, trashcan disabled.
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.Yes)
        gui.on_delete_calibration()
        assert not prof.exists()
        assert gui.calibration_select.currentText() == "None"
        assert not gui.delete_calibration_button.isEnabled()
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_field_edit_resets_saved_config_to_none(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        gui.calibration_select.addItem("some_prof")
        gui.calibration_select.setCurrentText("some_prof")
        # A user edit to a calibration field.
        gui.calib_fields["image_rot_deg"].setText("42.0")
        gui._on_calib_field_edited()
        assert gui.calibration_select.currentText() == "None"
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_calibration_stage_resets_saved_config_to_none(qapp):
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        gui.calibration_select.addItem("some_prof")
        gui.calibration_select.setCurrentText("some_prof")
        # A fit stage lands fresh parameters.
        gui.on_calibration_stage({
            "stage": "rotation", "params": {"image_rot_deg": 10.0},
            "preview": None, "reference": None})
        assert gui.calibration_select.currentText() == "None"
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_selecting_saved_config_does_not_reset(qapp, tmp_path):
    import yaml
    from fpwfsc.tokyo_drift.tokyo_drift_GUI import TokyoDriftConfigGUI

    gui = TokyoDriftConfigGUI()
    try:
        gui.calibrations_dir = tmp_path
        (tmp_path / "keepme.yaml").write_text(yaml.safe_dump(
            {"mode": "vampires_f760_10zern", "image_rot_deg": 3.0,
             "flip_x": True}))
        gui.populate_calibration_selector(select="keepme")
        # Loading the profile sets the fields but must NOT bounce to None.
        gui.on_calibration_selected("keepme")
        assert gui.calibration_select.currentText() == "keepme"
    finally:
        gui.thread_check_timer.stop()
        gui.close()


def test_calibration_reads_current_preset_not_stale(qapp, monkeypatch):
    """Regression: View/Auto-calibrate/Fine-tune must read the CURRENT
    form values. The preset dropdown change previously never reached the
    bench sim (the form was only flushed on Save/Run)."""
    import fpwfsc.tokyo_drift.tokyo_drift_GUI as G

    class _Sig:
        def connect(self, *_a):
            pass

    captured = {}

    class FakeThread:
        stage_update = _Sig()
        view_ready = _Sig()
        calibration_done = _Sig()
        calibration_failed = _Sig()

        def __init__(self, mode, preset, seed, **_kw):
            captured["preset"] = preset

        def start(self):
            pass

    gui = G.TokyoDriftConfigGUI()
    try:
        monkeypatch.setattr(G, "CalibrationThread", FakeThread)
        gui.preset_combo.setCurrentText("stress_test")  # user changes it
        gui._start_calibration_thread("view")
        assert captured["preset"] == "stress_test"  # not the stale default
    finally:
        gui.thread_check_timer.stop()
        gui.close()
