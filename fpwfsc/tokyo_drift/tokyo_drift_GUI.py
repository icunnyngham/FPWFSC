"""Tokyo Drift GUI.

PyQt5 config-editor GUI following the FPWFSC per-pipeline convention
(fnf/FF_GUI.py is the closest relative): an auto-rendered config form
backed by the .ini/.spec contract, with Run/Stop driving the pipeline's
``run()`` in a QThread.

tokyo_drift-specific: three selectors at the top instead of one —
Hardware (Sim/Vampires), Mode (which trained-model setup to run), and
Calibration (which saved calibration profile to apply; the list is
filtered to profiles fit against the selected mode). Mode and
Calibration write into the ``[MODE]`` config section, so a saved .ini
fully reproduces a GUI-configured run.
"""
import datetime
import os
import sys
import threading
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton,
                             QScrollArea, QFrame, QToolButton, QSizePolicy,
                             QFileDialog, QGridLayout, QInputDialog,
                             QMessageBox, QStyle)
from PyQt5.QtCore import (Qt, pyqtSignal, pyqtSlot, QParallelAnimationGroup,
                          QPropertyAnimation, QAbstractAnimation, QTimer,
                          QThread)
from PyQt5.QtGui import QFont

from configobj import ConfigObj, ConfigObjError, flatten_errors

# MyValidator adds the *_or_none types our spec uses on top of stock
# configobj validation.
from fpwfsc.common.support_functions import MyValidator as Validator

# Same-directory plotter: relative import with a script-mode fallback
# (mirrors fnf/FF_GUI.py). Modules with package-relative dependencies
# (run, gui_helper, common) must be imported absolutely — the installed
# fpwfsc package resolves them even when this file runs as a script.
try:
    from . import tokyo_drift_plotter_qt as pf
except ImportError:
    import tokyo_drift_plotter_qt as pf

from fpwfsc.tokyo_drift import gui_helper as helper
from fpwfsc.tokyo_drift.calibration.profiles import (PROFILE_DEFAULTS,
                                                     delete_profile,
                                                     load_profile,
                                                     save_profile)
from fpwfsc.tokyo_drift.preprocess import PreprocessImage
from fpwfsc.tokyo_drift.run import run

# Config sections owned by the top dropdowns rather than the form.
SELECTOR_SECTIONS = ("MODE",)

# Calibration-parameter fields shown in the collapsible workbench panel.
CALIB_FLOAT_FIELDS = ("image_rot_deg", "dm_scale", "dm_rot_deg")
CALIB_INT_FIELDS = ("crop_cx", "crop_cy")       # blank/None -> auto center
CALIB_BOOL_FIELDS = ("flip_x", "flip_y")
CALIB_FIELD_ORDER = ("image_rot_deg", "crop_cx", "crop_cy", "flip_x",
                     "flip_y", "dm_scale", "dm_rot_deg")


def diff_rms(frame, reference):
    """RMS of the peak-normalized difference image — the per-stage
    calibration-progress metric. None if shapes don't match (e.g. the
    raw probe frame before rotation/cropping)."""
    frame = np.asarray(frame, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if frame.shape != reference.shape:
        return None
    if frame.max() <= 0 or reference.max() <= 0:
        return None
    residual = frame / frame.max() - reference / reference.max()
    return float(np.sqrt(np.mean(residual ** 2)))


class CalibrationThread(QThread):
    """Runs calibration work off the GUI thread.

    Tasks: ``'view'`` acquires the probe frame + ideal reference and
    stops (manual-adjustment mode); ``'auto'`` runs the full staged
    fit; ``'fine'`` re-fits in a restricted range around the ``around``
    profile. Existing bench/ideal sims are reused when passed.
    """
    stage_update = pyqtSignal(dict)
    view_ready = pyqtSignal(dict)
    calibration_done = pyqtSignal(object, object, object, object)
    calibration_failed = pyqtSignal(str)

    def __init__(self, mode_name, preset, seed, task='auto', around=None,
                 bench=None, ideal=None, probe_amplitude=0.3):
        super().__init__()
        self.mode_name = mode_name
        self.preset = preset
        self.seed = seed
        self.task = task
        self.around = around
        self.bench = bench
        self.ideal = ideal
        self.probe_amplitude = probe_amplitude

    def run(self):
        try:
            from fpwfsc.tokyo_drift.calibration.harness import (
                acquire_probe,
                calibrate_bench_sim,
            )
            from fpwfsc.tokyo_drift.sim import BenchSim, IdealSim

            bench = self.bench or BenchSim.from_mode(
                self.mode_name, preset=self.preset, seed=self.seed)
            ideal = self.ideal or IdealSim.from_mode(self.mode_name)

            if self.task == 'view':
                ctx = acquire_probe(self.mode_name, bench=bench,
                                    ideal=ideal,
                                    probe_amplitude=self.probe_amplitude)
                self.view_ready.emit(ctx)
                return

            profile, report = calibrate_bench_sim(
                self.mode_name, bench=bench, ideal=ideal,
                stage_callback=self.stage_update.emit,
                around=self.around if self.task == 'fine' else None,
                probe_amplitude=self.probe_amplitude)
            self.calibration_done.emit(profile, report, bench, ideal)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.calibration_failed.emit(str(exc))


class AlgorithmThread(QThread):
    def __init__(self, camera, aosystem, config, spec_file, my_event, plotter):
        super().__init__()
        self.camera = camera
        self.aosystem = aosystem
        self.config = config
        self.spec_file = spec_file
        self.my_event = my_event
        self.plotter = plotter

    def run(self):
        run(camera=self.camera,
            aosystem=self.aosystem,
            config=self.config,
            configspec=self.spec_file,
            my_event=self.my_event,
            plotter=self.plotter)


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.content_area = QScrollArea()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setWidgetResizable(True)
        self.content_area.setFrameShape(QFrame.NoFrame)

        self.content_widget = QWidget()
        self.content_area.setWidget(self.content_widget)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation = QParallelAnimationGroup(self)
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"minimumHeight"))
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"maximumHeight"))
        self.toggle_animation.addAnimation(QPropertyAnimation(self.content_area, b"maximumHeight"))

    @pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if not checked else Qt.RightArrow)
        self.toggle_animation.setDirection(QAbstractAnimation.Forward if not checked else QAbstractAnimation.Backward)
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        self.content_widget.setLayout(layout)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()

        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(300)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        content_animation.setDuration(300)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class TokyoDriftConfigGUI(QWidget):
    def __init__(self):
        super().__init__()

        script_dir = Path(__file__).parent
        config_path = script_dir / "tokyo_drift_config_sim.ini"
        spec_path = script_dir / "tokyo_drift_config.spec"

        self.config_file = str(config_path)
        self.spec_file = str(spec_path)
        self.is_running = False
        self.my_event = threading.Event()

        # Calibration-workbench state (populated by a calibration run)
        self.calibrations_dir = helper.CALIBRATIONS_DIR
        self.calib_fields = {}
        self._updating_fields = False
        self.calib_plotter = None
        self.calib_raw = None          # cached raw probe frame
        self.calib_ref = None          # ideal probe reference
        self.calib_ideal = None        # IdealSim for dm_scale re-renders
        self.calib_bench = None
        self.calib_probe = None
        self.calib_corrector = None
        self._last_scale_render = None
        self._calib_progress = {"x": [], "y": [], "stage_marks": []}
        self._calib_context_key = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tokyo Drift')
        self.resize(425, 875)

        main_layout = QVBoxLayout(self)

        # --- Top selectors: Hardware / Mode ----------------------------
        # (Saved-config selection + the calibration actions live in the
        # Model<->instrument alignment section below.)
        selector_layout = QGridLayout()
        selector_layout.setVerticalSpacing(2)

        selector_layout.addWidget(QLabel("Hardware"), 0, 0)
        self.hardware_select = QComboBox()
        self.hardware_select.addItems(helper.valid_instruments)
        self.hardware_select.currentTextChanged.connect(self.on_hardware_changed)
        self.hardware_select.setFixedHeight(20)
        selector_layout.addWidget(self.hardware_select, 0, 1)

        selector_layout.addWidget(QLabel("Mode"), 1, 0)
        self.mode_select = QComboBox()
        self.mode_select.setFixedHeight(20)
        self.mode_select.currentTextChanged.connect(self.on_mode_changed)
        selector_layout.addWidget(self.mode_select, 1, 1)

        main_layout.addLayout(selector_layout)

        # --- Model<->instrument alignment section ----------------------
        # Groups the sim-only bench preset, the calibration probe
        # amplitude, and the fitted-parameter panel together (top to
        # bottom), above the auto-rendered config form. Hand-built: the
        # ALIGNMENT config section is excluded from the auto-form so these
        # controls can sit alongside the fitted-calibration panel; their
        # values round-trip through _sync_alignment_widgets_{from,to}_config.
        align_frame = QFrame()
        align_frame.setFrameShape(QFrame.StyledPanel)
        align_outer = QVBoxLayout(align_frame)
        align_outer.setSpacing(4)
        align_outer.addWidget(QLabel(f"<b>{helper.ALIGNMENT_DISPLAY}</b>"))

        # bench sim preset (SIM ONLY): registry-backed dropdown, hidden
        # when the hardware selector is not 'Sim'. Values populated after
        # the config loads.
        self.preset_row = QWidget()
        preset_layout = QHBoxLayout(self.preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_label = QLabel("bench sim preset")
        preset_label.setToolTip(
            helper.get_help_message(helper.ALIGNMENT_SECTION, "bench sim preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            str(c) for c in
            (helper.get_choices(helper.ALIGNMENT_SECTION, "bench sim preset") or [])])
        self.preset_combo.setFixedHeight(20)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        align_outer.addWidget(self.preset_row)

        # probe amplitude (used in both sim and real calibration).
        amp_row = QWidget()
        amp_layout = QHBoxLayout(amp_row)
        amp_layout.setContentsMargins(0, 0, 0, 0)
        amp_label = QLabel("probe amplitude")
        amp_label.setToolTip(
            helper.get_help_message(helper.ALIGNMENT_SECTION, "probe amplitude"))
        self.probe_amp_field = QLineEdit("")
        self.probe_amp_field.setFixedHeight(20)
        amp_layout.addWidget(amp_label)
        amp_layout.addWidget(self.probe_amp_field)
        align_outer.addWidget(amp_row)

        # Collapsible fitted-calibration panel: populated by the staged
        # fit, hand-editable with live panel re-rendering.
        self.calib_box = CollapsibleBox("Calibration parameters")
        calib_layout = QGridLayout()
        calib_layout.setVerticalSpacing(2)
        calib_layout.setHorizontalSpacing(5)
        for i, key in enumerate(CALIB_FIELD_ORDER):
            label = QLabel(key)
            if key in CALIB_BOOL_FIELDS:
                widget = QComboBox()
                widget.addItems(['False', 'True'])
                widget.currentTextChanged.connect(self._on_calib_field_edited)
            else:
                widget = QLineEdit("")
                widget.editingFinished.connect(self._on_calib_field_edited)
            widget.setFixedHeight(20)
            calib_layout.addWidget(label, i, 0)
            calib_layout.addWidget(widget, i, 1)
            self.calib_fields[key] = widget
        self.calib_box.setContentLayout(calib_layout)
        align_outer.addWidget(self.calib_box)

        # Saved-config row: pick / delete / save a named calibration.
        saved_row = QHBoxLayout()
        saved_row.addWidget(QLabel("Saved Config"))
        self.calibration_select = QComboBox()
        self.calibration_select.setFixedHeight(20)
        self.calibration_select.currentTextChanged.connect(
            self.on_calibration_selected)
        self.calibration_select.currentTextChanged.connect(
            lambda _=None: self._update_delete_button_state())
        saved_row.addWidget(self.calibration_select, 1)

        self.delete_calibration_button = QPushButton()
        self.delete_calibration_button.setIcon(
            self.style().standardIcon(QStyle.SP_TrashIcon))
        self.delete_calibration_button.setToolTip(
            "Delete the selected saved config")
        self.delete_calibration_button.setFixedSize(26, 22)
        self.delete_calibration_button.clicked.connect(
            self.on_delete_calibration)
        saved_row.addWidget(self.delete_calibration_button)

        self.save_calibration_button = QPushButton('Save')
        self.save_calibration_button.setFixedHeight(22)
        self.save_calibration_button.clicked.connect(self.on_save_calibration)
        saved_row.addWidget(self.save_calibration_button)
        align_outer.addLayout(saved_row)

        # Calibration actions. View: acquire + display, manual adjustment
        # only. Auto-calibrate: full staged fit (global rotation search).
        # Fine tune: restricted re-fit around the current field values.
        calib_buttons = QHBoxLayout()
        self.view_button = QPushButton('View')
        self.view_button.clicked.connect(
            lambda: self._start_calibration_thread('view'))
        calib_buttons.addWidget(self.view_button)
        self.calibrate_button = QPushButton('Auto-calibrate')
        self.calibrate_button.clicked.connect(
            lambda: self._start_calibration_thread('auto'))
        calib_buttons.addWidget(self.calibrate_button)
        self.fine_tune_button = QPushButton('Fine tune')
        self.fine_tune_button.clicked.connect(
            lambda: self._start_calibration_thread('fine'))
        calib_buttons.addWidget(self.fine_tune_button)
        align_outer.addLayout(calib_buttons)

        main_layout.addWidget(align_frame)
        # Start from the profile defaults, not blank fields
        self._set_calibration_fields(PROFILE_DEFAULTS)
        self._update_delete_button_state()

        # --- Scrollable auto-rendered config form ----------------------
        scroll = QScrollArea(self)
        main_layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget(scroll)

        self.layout = QVBoxLayout(scroll_content)
        self.layout.setSpacing(2)
        scroll.setWidget(scroll_content)

        self.load_config(initial_load=True)
        self.populate_mode_selectors()
        self.create_widgets()
        self._sync_alignment_widgets_from_config()

        # --- Buttons ----------------------------------------------------
        button_layout = QGridLayout()
        button_layout.setVerticalSpacing(2)
        button_layout.setHorizontalSpacing(5)

        self.run_stop_button = QPushButton('Run')
        self.run_stop_button.setFont(QFont('Arial', 10, QFont.Bold))
        self.run_stop_button.clicked.connect(self.toggle_run_stop)
        self.run_stop_button.setStyleSheet("background-color: green; color: white;")
        button_layout.addWidget(self.run_stop_button, 0, 0)

        save_button = QPushButton('Save configuration')
        save_button.clicked.connect(self.save_config)
        button_layout.addWidget(save_button, 0, 1)

        load_config_button = QPushButton('Load configuration')
        load_config_button.clicked.connect(lambda: self.load_config(None))
        button_layout.addWidget(load_config_button, 1, 0)

        main_layout.addLayout(button_layout)

        self.thread_check_timer = QTimer()
        self.thread_check_timer.timeout.connect(self.check_thread_status)
        self.thread_check_timer.start(1000)

        self.default_hardware = helper.valid_instruments[0]
        self.on_hardware_changed(self.default_hardware)

    # --- Mode / Calibration selectors -----------------------------------

    def populate_mode_selectors(self):
        """Fill the Mode dropdown from the on-disk registry and sync the
        Calibration dropdown to the selected mode."""
        modes = helper.list_modes()
        config_mode = self.config['MODE']['mode name']
        if config_mode and config_mode not in modes:
            # Keep the configured mode selectable even before its
            # registry directory exists.
            modes = [config_mode] + modes

        self.mode_select.blockSignals(True)
        self.mode_select.clear()
        self.mode_select.addItems(modes)
        if config_mode:
            index = self.mode_select.findText(str(config_mode))
            if index >= 0:
                self.mode_select.setCurrentIndex(index)
        self.mode_select.blockSignals(False)

        self.populate_calibration_selector()

    def populate_calibration_selector(self, select=None):
        """Calibration choices are filtered to the selected mode."""
        mode = self.mode_select.currentText()
        profiles = ['None'] + helper.list_calibrations(
            mode_name=mode or None, calibrations_dir=self.calibrations_dir)

        current = select or str(self.config['MODE']['calibration profile'])
        self.calibration_select.blockSignals(True)
        self.calibration_select.clear()
        self.calibration_select.addItems(profiles)
        index = self.calibration_select.findText(current)
        if index >= 0:
            self.calibration_select.setCurrentIndex(index)
        self.calibration_select.blockSignals(False)
        # Signals were blocked above, so refresh the trashcan explicitly.
        self._update_delete_button_state()

    def on_mode_changed(self, _mode_name):
        self.populate_calibration_selector()
        self._warn_if_model_unavailable()

    def _model_availability_message(self, mode):
        """None if the mode's NN checkpoint resolves, else an explanatory
        message. Pure logic (no UI) so it can be unit-tested."""
        if not mode:
            return None
        from fpwfsc.tokyo_drift.mode_registry import checkpoint_path
        try:
            checkpoint_path(mode)
            return None
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return (
                f"The trained NN for mode '{mode}' is not available:\n\n"
                f"{exc}\n\nThe 'Tokyo Drift (NN)' predictor will not run "
                f"until the checkpoint is in place; the debug predictors "
                f"(Oracle / Random walk) still work.")

    def _warn_if_model_unavailable(self):
        """Alert on mode entry if the NN checkpoint is missing, so the
        user learns before Run rather than mid-loop."""
        message = self._model_availability_message(
            self.mode_select.currentText())
        if message is None:
            return
        print(f"tokyo_drift: {message}")
        # A modal would block headless (offscreen) test runs; skip it there.
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QMessageBox.warning(self, "Model checkpoint unavailable", message)

    def on_calibration_selected(self, name):
        """Load a saved profile into the parameter panel (and preview it
        live if a calibration frame is cached)."""
        if not name or name == 'None':
            return
        try:
            profile = load_profile(name, calibrations_dir=self.calibrations_dir)
        except FileNotFoundError as exc:
            print(f"Error loading calibration {name!r}: {exc}")
            return
        self._set_calibration_fields(profile)
        self.refresh_calibration_preview()

    def _update_delete_button_state(self):
        """The trashcan is enabled only when a named config is selected."""
        name = self.calibration_select.currentText()
        self.delete_calibration_button.setEnabled(bool(name) and name != 'None')

    def _mark_config_unsaved(self):
        """Point the Saved Config selector back at 'None': the displayed
        parameters no longer match a saved profile (they were edited or
        freshly calibrated). Signals are blocked so this does not reload."""
        if self.calibration_select.currentText() == 'None':
            return
        self.calibration_select.blockSignals(True)
        index = self.calibration_select.findText('None')
        if index >= 0:
            self.calibration_select.setCurrentIndex(index)
        self.calibration_select.blockSignals(False)
        self._update_delete_button_state()

    def _on_calib_field_edited(self, *_args):
        """A user edit to a calibration field: the parameters no longer
        match the saved profile, so drop back to 'None' and re-preview.
        Programmatic updates (loading a profile, streaming a fit) set
        ``_updating_fields`` and are ignored here."""
        if self._updating_fields:
            return
        self._mark_config_unsaved()
        self.refresh_calibration_preview()

    def on_delete_calibration(self):
        """Delete the selected saved config after a confirmation prompt."""
        name = self.calibration_select.currentText()
        if not name or name == 'None':
            return
        reply = QMessageBox.question(
            self, "Delete saved config",
            f"Delete the saved configuration '{name}'?\n"
            f"This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            delete_profile(name, calibrations_dir=self.calibrations_dir)
        except FileNotFoundError as exc:
            print(f"Delete failed: {exc}")
        # Refresh the list (drops the deleted entry) and land on 'None'.
        self.populate_calibration_selector(select='None')
        self._update_delete_button_state()
        print(f"Deleted saved config '{name}'.")

    # --- Calibration workbench -----------------------------------------

    def _set_calib_buttons_enabled(self, enabled, busy_label=None):
        for button, label in ((self.view_button, 'View'),
                              (self.calibrate_button, 'Auto-calibrate'),
                              (self.fine_tune_button, 'Fine tune')):
            button.setEnabled(enabled)
            button.setText(label if enabled or busy_label is None
                           else busy_label)

    def _start_calibration_thread(self, task):
        if self.hardware_select.currentText() != 'Sim':
            print("Calibration: real-hardware calibration is not wired "
                  "yet; select 'Sim'.")
            return
        around = None
        if task == 'fine':
            try:
                around = self._profile_from_fields()
            except ValueError as exc:
                print(f"Fine tune: bad field value ({exc})")
                return
        if self.calib_plotter is None or self.calib_plotter.closed:
            self.calib_plotter = pf.LivePlotter()
        self._set_calib_buttons_enabled(False, busy_label='Working...')
        self._calib_progress = {"x": [], "y": [], "stage_marks": []}

        # Sync the form into self.config first: View / Auto-calibrate /
        # Fine tune all read the preset, seed, and probe amplitude from
        # self.config, which otherwise still holds the loaded values (edits
        # to the dropdowns/fields were never flushed) — the bench sim would
        # silently ignore a changed preset.
        self.update_config_from_gui()

        mode = self.mode_select.currentText()
        preset = str(self.config['ALIGNMENT']['bench sim preset'])
        seed = self.config['SIMULATION']['seed']
        seed = None if seed in (None, 'None', '') else int(seed)

        probe_amplitude = float(
            self.config['ALIGNMENT']['probe amplitude'])
        self._current_probe_amplitude = probe_amplitude

        # Reuse the built sims only while mode/preset/seed are unchanged
        context_key = (mode, preset, seed)
        bench = ideal = None
        if context_key == getattr(self, '_calib_context_key', None):
            bench, ideal = self.calib_bench, self.calib_ideal
        self._calib_context_key = context_key

        self.calibration_thread = CalibrationThread(
            mode, preset, seed, task=task, around=around,
            bench=bench, ideal=ideal, probe_amplitude=probe_amplitude)
        self.calibration_thread.stage_update.connect(self.on_calibration_stage)
        self.calibration_thread.view_ready.connect(self.on_view_ready)
        self.calibration_thread.calibration_done.connect(self.on_calibration_done)
        self.calibration_thread.calibration_failed.connect(self.on_calibration_failed)
        self.calibration_thread.start()

    def on_view_ready(self, ctx):
        """View acquisition finished: display raw + ideal, enable
        manual adjustment against the cached frame."""
        self._set_calib_buttons_enabled(True)
        self.calib_bench = ctx["bench"]
        self.calib_ideal = ctx["ideal"]
        self.calib_raw = ctx["raw"]
        self.calib_ref = ctx["reference"]
        self.calib_probe = ctx["probe"]
        self.calib_corrector = ctx["corrector"]
        self._last_scale_render = None
        if self.calib_plotter is not None and not self.calib_plotter.closed:
            amp = ctx.get("probe_amplitude")
            self.calib_plotter.update({
                "source": ctx["raw"],
                "ideal": ctx["reference"],
                "source_title": "View: raw detector frame (probe poked)",
                "ideal_title": f"Ideal probe (coma+trefoil, amp {amp})",
            })
        print("View ready: raw probe frame and ideal reference displayed. "
              "Adjust the calibration parameters to align them, or run "
              "Auto-calibrate / Fine tune.")

    def on_calibration_stage(self, payload):
        print(f"Calibration stage: {payload['stage']} -> {payload['params']}")
        self._set_calibration_fields(payload['params'])
        # A fit is producing fresh parameters -> no longer a saved config.
        self._mark_config_unsaved()
        if self.calib_plotter is None or self.calib_plotter.closed:
            return

        title = f"Calibration: {payload['stage']}"
        rms = diff_rms(payload["preview"], payload["reference"])
        if rms is not None:
            # One point per stage; dashed annotated marker at each.
            step = len(self._calib_progress["x"])
            self._calib_progress["x"].append(step)
            self._calib_progress["y"].append(rms)
            self._calib_progress["stage_marks"].append(
                (step, payload["stage"]))
            title += f" (RMS diff {rms:.4f})"

        amp = getattr(self, '_current_probe_amplitude', None)
        update = {
            "source": payload["preview"],
            "ideal": payload["reference"],
            "source_title": title,
            "ideal_title": f"Ideal probe (coma+trefoil, amp {amp})",
            "progress": {
                "x": list(self._calib_progress["x"]),
                "y": list(self._calib_progress["y"]),
                "stage_marks": list(self._calib_progress["stage_marks"]),
                "title": "Calibration progress",
                "ylabel": "RMS(diff)",
            },
        }
        if payload.get("curve") is not None:
            update["curve"] = payload["curve"]
            update["curve_title"] = payload.get("curve_title", "Score")
        self.calib_plotter.update(update)

    def on_calibration_done(self, profile, report, bench, ideal):
        self._set_calib_buttons_enabled(True)
        self.calib_bench = bench
        self.calib_raw = report["stage_previews"]["raw"]
        self.calib_ref = report["reference_psf"]
        self.calib_ideal = ideal
        self.calib_probe = report["probe_coefficients"]
        self.calib_corrector = report["corrector"]
        self._last_scale_render = None
        self._set_calibration_fields(profile)
        print("Calibration complete. Sim-only recovery report: "
              f"rotation error {report['image_rot_error_deg']:.3f} deg, "
              f"fitted/injected scale {report['dm_scale_over_truth']:.3f} "
              "(sits above 1 by the DM influence-function gain - that IS "
              "the effective gain the loop needs), "
              f"flips-as-expected {report['flips_expected_false']}.")
        print("Review/edit the parameters, then 'Save calibration'.")

    def on_calibration_failed(self, message):
        self._set_calib_buttons_enabled(True)
        print(f"Calibration failed: {message}")

    def _set_calibration_fields(self, params):
        self._updating_fields = True
        try:
            for key, value in params.items():
                widget = self.calib_fields.get(key)
                if widget is None:
                    continue
                if key in CALIB_BOOL_FIELDS:
                    widget.setCurrentText(str(bool(value)))
                else:
                    # 'None' (auto) shown explicitly rather than blank
                    widget.setText("None" if value is None else str(value))
        finally:
            self._updating_fields = False

    def _profile_from_fields(self):
        """Parse the parameter panel into a profile dict (ValueError on
        malformed entries)."""
        profile = {}
        for key, widget in self.calib_fields.items():
            if key in CALIB_BOOL_FIELDS:
                profile[key] = widget.currentText() == 'True'
                continue
            text = widget.text().strip()
            if key in CALIB_INT_FIELDS:
                profile[key] = None if text in ("", "None") else int(float(text))
            else:
                profile[key] = float(text) if text else 0.0
        return profile

    def refresh_calibration_preview(self, *_args):
        """Re-render the alignment panels from the cached calibration
        frame using the (possibly hand-edited) parameter fields."""
        if self._updating_fields:
            return
        if (self.calib_raw is None or self.calib_plotter is None
                or self.calib_plotter.closed):
            return
        try:
            profile = self._profile_from_fields()
        except ValueError as exc:
            print(f"Calibration preview: bad field value ({exc})")
            return

        preprocess = PreprocessImage(
            crop_res=np.asarray(self.calib_ref).shape[0],
            rot_angle=profile["image_rot_deg"],
            center_x=profile["crop_cx"], center_y=profile["crop_cy"],
            flip_horizontal=profile["flip_x"],
            flip_vertical=profile["flip_y"], verbose=False)
        frame = preprocess.process(self.calib_raw, normalize=False)

        ideal_img = self.calib_ref
        # dm_scale edits re-render the ideal probe at the new amplitude
        # (one cheap ideal-sim sample; no new bench exposure).
        if (self.calib_ideal is not None
                and abs(profile["dm_scale"] - 1.0) > 1e-9):
            if (self._last_scale_render is None
                    or self._last_scale_render[0] != profile["dm_scale"]):
                rendered = self.calib_ideal.psf(
                    {self.calib_corrector:
                     np.asarray(self.calib_probe) * profile["dm_scale"]})
                self._last_scale_render = (profile["dm_scale"], rendered)
            ideal_img = self._last_scale_render[1]

        title = "Calibration: manual edit"
        rms = diff_rms(frame, ideal_img)
        if rms is not None:
            title += f" (RMS diff {rms:.4f})"
        self.calib_plotter.update({
            "source": frame,
            "ideal": ideal_img,
            "source_title": title,
        })

    def on_save_calibration(self):
        try:
            profile = self._profile_from_fields()
        except ValueError as exc:
            print(f"Save calibration: bad field value ({exc})")
            return
        mode = self.mode_select.currentText()
        profile["mode"] = mode
        profile.setdefault("shift_x", 0)
        profile.setdefault("shift_y", 0)

        default = f"{mode}_{datetime.date.today().isoformat()}"
        name, ok = QInputDialog.getText(self, "Save calibration",
                                        "Profile name:", text=default)
        if not (ok and name):
            return
        path = save_profile(name, profile,
                            calibrations_dir=self.calibrations_dir)
        print(f"Calibration saved to {path}")
        self.populate_calibration_selector(select=name)
        self.config['MODE']['calibration profile'] = name

    def on_hardware_changed(self, selected_hardware):
        try:
            print(f"Loading {selected_hardware}...")
            self.camera, self.aosystem = helper.load_instruments(
                selected_hardware, camargs={}, aoargs={})
            print(f"{selected_hardware} loaded successfully")
        except Exception as e:
            print(f"Error loading {selected_hardware}: {str(e)}")
            print("Loading Sim as default")
            default_index = self.hardware_select.findText('Sim')
            if default_index >= 0:
                self.hardware_select.blockSignals(True)
                self.hardware_select.setCurrentIndex(default_index)
                self.hardware_select.blockSignals(False)
                self.camera, self.aosystem = 'Sim', 'Sim'
        self._apply_sim_only_visibility()

    def _apply_sim_only_visibility(self):
        """Show sim-only controls (the bench-sim preset) only in Sim mode;
        they inject fake hardware and have no meaning on a real instrument."""
        is_sim = self.hardware_select.currentText() == 'Sim'
        self.preset_row.setVisible(is_sim)

    # --- Run / stop -------------------------------------------------------

    def check_thread_status(self):
        if hasattr(self, 'algorithm_thread'):
            if not self.algorithm_thread.isRunning() and self.is_running:
                self.is_running = False
                self.run_stop_button.setText('Run')
                self.run_stop_button.setStyleSheet("background-color: green; color: white;")

    def toggle_run_stop(self):
        if not self.is_running:
            self.is_running = True
            self.run_stop_button.setText('Stop')
            self.run_stop_button.setStyleSheet("background-color: red; color: white;")
            print("Running")
            self.my_event = threading.Event()
            self.run_config()
        else:
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet("background-color: green; color: white;")
            print("Stopping")
            if hasattr(self, 'my_event'):
                self.my_event.set()
                self.algorithm_thread.wait()

    def run_config(self):
        self.update_config_from_gui()
        validator = Validator()
        results = self.config.validate(validator, preserve_errors=True)
        if results is not True:
            error_messages = []
            for section_list, key, error in flatten_errors(self.config, results):
                if key is not None:
                    section_str = '.'.join(section_list)
                    error_messages.append(f"Invalid value in [{section_str}] {key}: {str(error)}")
            print("\nConfiguration Validation Failed:")
            for msg in error_messages:
                print(f"  {msg}")
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet("background-color: green; color: white;")
            return

        # The NN predictor needs its checkpoint present: fail fast with an
        # informative dialog rather than a mid-run thread traceback.
        if self.config['LOOP_SETTINGS']['predictor'] == 'model':
            message = self._model_availability_message(
                self.config['MODE']['mode name'])
            if message is not None:
                print(f"tokyo_drift: cannot run - {message}")
                if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                    QMessageBox.critical(
                        self, "Cannot run: model unavailable", message)
                self.is_running = False
                self.run_stop_button.setText('Run')
                self.run_stop_button.setStyleSheet(
                    "background-color: green; color: white;")
                return

        # The loop only runs against a calibration profile. If none is
        # selected, offer to save the current workbench parameters or
        # run with them unsaved (via an ephemeral profile file).
        resolved_profile = self._resolve_run_profile()
        if resolved_profile is None:
            self.is_running = False
            self.run_stop_button.setText('Run')
            self.run_stop_button.setStyleSheet(
                "background-color: green; color: white;")
            return

        print("Current configuration:")
        for section in self.config.sections:
            print(f"[{section}]")
            for key, value in self.config[section].items():
                print(f"    {key} = {value}")
            print()

        if self.config['LOOP_SETTINGS']['Plot']:
            self.plotter = pf.LivePlotter()
        else:
            self.plotter = None
        self.my_event = threading.Event()

        # The thread gets its own config copy (with the resolved
        # profile), so GUI edits mid-run never race the loop.
        thread_config = ConfigObj(self.config)
        thread_config['MODE']['calibration profile'] = resolved_profile

        self.algorithm_thread = AlgorithmThread(
            camera=self.camera,
            aosystem=self.aosystem,
            config=thread_config,
            spec_file=self.spec_file,
            my_event=self.my_event,
            plotter=self.plotter
        )
        self.algorithm_thread.start()

    def _resolve_run_profile(self):
        """The calibration profile the loop should run with, or None to
        abort. A selected named profile passes straight through; with
        none selected, the user chooses: save the current parameters
        first, run with them unsaved, or cancel."""
        name = self.calibration_select.currentText()
        if name and name != 'None':
            return name

        choice = self._ask_unsaved_run()
        if choice == 'cancel':
            print("Run cancelled: no calibration profile.")
            return None
        if choice == 'save':
            self.on_save_calibration()
            name = self.calibration_select.currentText()
            if name and name != 'None':
                return name
            print("Run cancelled: calibration was not saved.")
            return None

        # Run without saving: ephemeral profile from the current fields
        import tempfile

        import yaml
        try:
            profile = self._profile_from_fields()
        except ValueError as exc:
            print(f"Run cancelled: bad calibration field value ({exc})")
            return None
        profile["mode"] = self.mode_select.currentText()
        profile.setdefault("shift_x", 0)
        profile.setdefault("shift_y", 0)
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", prefix="tokyo_drift_unsaved_",
            delete=False)
        yaml.safe_dump(profile, tmp)
        tmp.close()
        print(f"Running with unsaved calibration parameters "
              f"(ephemeral profile {tmp.name})")
        return tmp.name

    def _ask_unsaved_run(self):
        """Dialog: 'save' / 'run' (without saving) / 'cancel'."""
        box = QMessageBox(self)
        box.setWindowTitle("No saved calibration")
        box.setText("No calibration profile is selected.\n"
                    "The loop needs alignment parameters to run.")
        save_button = box.addButton("Save current first...",
                                    QMessageBox.AcceptRole)
        run_button = box.addButton("Run without saving",
                                   QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == save_button:
            return 'save'
        if clicked == run_button:
            return 'run'
        return 'cancel'

    # --- Config load / save -------------------------------------------

    def load_config(self, file_name=None, initial_load=False):
        """Load a configuration file and update the GUI.

        If initial_load is True, use the default config file. If
        file_name is None and not initial_load, open a file dialog.
        """
        if initial_load:
            file_name = self.config_file
        elif file_name is None:
            dialog = QFileDialog(self)
            dialog.setNameFilter("INI Files (*.ini)")
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setViewMode(QFileDialog.List)

            if dialog.exec_():
                file_name = dialog.selectedFiles()[0]
            else:
                print("No file selected.")
                return

        try:
            new_config = ConfigObj(file_name, configspec=self.spec_file)
            validator = Validator()
            results = new_config.validate(validator, preserve_errors=True)

            if results is not True:
                error_messages = []
                for section_list, key, error in flatten_errors(new_config, results):
                    if key is not None:
                        section_str = '.'.join(section_list)
                        if error is False:
                            error_messages.append(f"Missing value or section for '{section_str}.{key}'")
                        else:
                            error_messages.append(f"Invalid value for '{section_str}.{key}': {error}")
                    else:
                        error_messages.append(f"Missing section: {'.'.join(section_list)}")

                if error_messages:
                    print("Configuration validation failed:")
                    for msg in error_messages:
                        print(f"  - {msg}")
                    raise ConfigObjError("Config validation failed")

            self.config = new_config
            if not initial_load:
                self.populate_mode_selectors()
                self.update_gui_from_config()
            print(f"Configuration loaded successfully from {file_name}")
        except (IOError, ConfigObjError) as e:
            print(f"Error loading configuration: {e}")
            if initial_load:
                print("Error loading default configuration. Exiting.")
                sys.exit(1)

    def save_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        default_name = "tokyo_drift_config.ini"
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            default_name,
            "INI Files (*.ini);;All Files (*)",
            options=options
        )
        if file_name:
            if not file_name.lower().endswith('.ini'):
                file_name += '.ini'
            self.update_config_from_gui()
            self.config.filename = file_name
            self.config.write()
            print(f"Configuration saved successfully to {file_name}")

    # --- Form rendering (auto-generated from config + spec) -------------

    def create_widgets(self):
        for section, items in self.config.items():
            if (section in SELECTOR_SECTIONS
                    or section == helper.ALIGNMENT_SECTION):
                continue  # owned by the top dropdowns / hand-rendered above
            section_frame = QFrame()
            section_frame.setFrameShape(QFrame.StyledPanel)
            section_layout = QVBoxLayout(section_frame)
            section_layout.setSpacing(4)

            section_label = QLabel(f"<b>{helper.section_display_name(section)}</b>")
            section_layout.addWidget(section_label)

            regular_options = []
            expert_options = []

            for key, value in items.items():
                if helper.is_expert_option(section, key):
                    expert_options.append((key, value))
                else:
                    regular_options.append((key, value))

            for key, value in regular_options:
                item_widget = self.create_item_widget(key, value, section)
                section_layout.addWidget(item_widget)

            if expert_options:
                expert_box = CollapsibleBox("Expert Options")
                expert_layout = QGridLayout()
                expert_layout.setVerticalSpacing(2)
                expert_layout.setHorizontalSpacing(5)
                for i, (key, value) in enumerate(expert_options):
                    label = QLabel(key)
                    input_widget = self.create_input_widget(section, key, value)
                    expert_layout.addWidget(label, i, 0)
                    expert_layout.addWidget(input_widget, i, 1)

                    tooltip = helper.get_help_message(section, key)
                    label.setToolTip(tooltip)
                    input_widget.setToolTip(tooltip)

                expert_box.setContentLayout(expert_layout)
                section_layout.addWidget(expert_box)

            self.layout.addWidget(section_frame)

    def create_item_widget(self, key, value, section):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(key)
        layout.addWidget(label)

        input_widget = self.create_input_widget(section, key, value)
        layout.addWidget(input_widget)

        tooltip = helper.get_help_message(section, key)
        label.setToolTip(tooltip)
        input_widget.setToolTip(tooltip)

        return widget

    def create_input_widget(self, section, key, value):
        if helper.is_file_option(section, key):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            text_field = QLineEdit(str(value))
            text_field.setFixedHeight(20)

            browse_button = QPushButton("...")
            browse_button.setFixedWidth(30)
            browse_button.setFixedHeight(20)

            def browse_file(tf=text_field):
                file_name, _ = QFileDialog.getOpenFileName(
                    self, "Select File", tf.text(),
                    "FITS Files (*.fits *.fit);;All Files (*)"
                )
                if file_name:
                    tf.setText(file_name)

            browse_button.clicked.connect(lambda _, tf=text_field: browse_file(tf))
            layout.addWidget(text_field, 1)
            layout.addWidget(browse_button, 0)
            widget.setFixedHeight(24)
            widget.text_field = text_field
            return widget

        if helper.is_directory_option(section, key):
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            text_field = QLineEdit(str(value))
            text_field.setFixedHeight(20)

            browse_button = QPushButton("...")
            browse_button.setFixedWidth(30)
            browse_button.setFixedHeight(20)

            def browse_directory():
                directory = QFileDialog.getExistingDirectory(
                    self, "Select Directory", text_field.text(),
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
                )
                if directory:
                    text_field.setText(directory)

            browse_button.clicked.connect(lambda _: browse_directory())

            layout.addWidget(text_field, 1)
            layout.addWidget(browse_button, 0)

            widget.setFixedHeight(24)
            widget.text_field = text_field

            return widget

        # Registry-backed dropdowns (e.g. bench-sim presets) declared
        # via gui_helper config_info "choices"
        choices = helper.get_choices(section, key)
        if choices:
            input_widget = QComboBox()
            input_widget.addItems([str(c) for c in choices])
            input_widget.setCurrentText(str(value).strip())
            input_widget.setFixedHeight(20)
            return input_widget

        spec = self.get_spec_for_key(f"{section}.{key}")

        if spec:
            if 'option(' in spec:
                input_widget = QComboBox()
                labels = helper.get_option_labels(section, key)
                if labels:
                    # Explicit (value -> display label) map, display order;
                    # the stored value stays the internal token via itemData.
                    for opt_value, opt_label in labels:
                        input_widget.addItem(opt_label, opt_value)
                    index = input_widget.findData(str(value).strip())
                    if index >= 0:
                        input_widget.setCurrentIndex(index)
                    input_widget._uses_item_data = True
                else:
                    raw = spec.split('option(')[1].rsplit(')', 1)[0]
                    options = [o.strip().strip("'\"") for o in raw.split(',')]
                    # Drop the trailing default=... keyword the spec carries.
                    options = [o for o in options if o and '=' not in o]
                    input_widget.addItems(options)
                    input_widget.setCurrentText(str(value).strip())
            elif 'boolean' in spec:
                input_widget = QComboBox()
                input_widget.addItems(['True', 'False'])
                input_widget.setCurrentText(str(value))
            else:
                input_widget = QLineEdit(str(value))
        else:
            input_widget = QLineEdit(str(value))

        input_widget.setFixedHeight(20)
        return input_widget

    def get_spec_for_key(self, key):
        keys = key.split('.')
        spec = self.config.configspec
        for k in keys:
            if k in spec:
                spec = spec[k]
            else:
                return None
        return spec

    # --- GUI <-> config sync ---------------------------------------------

    def update_config_from_gui(self):
        # Selector-owned + hand-rendered sections first (not in the
        # auto-rendered scroll layout below).
        self.config['MODE']['mode name'] = self.mode_select.currentText()
        self.config['MODE']['calibration profile'] = self.calibration_select.currentText()
        self.config['ALIGNMENT']['bench sim preset'] = self.preset_combo.currentText()
        self.config['ALIGNMENT']['probe amplitude'] = self.probe_amp_field.text()

        for i in range(self.layout.count()):
            section_frame = self.layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text().strip('<b>').strip('</b>')
                section = helper.section_key_from_label(section)

                for j in range(1, section_layout.count()):
                    item = section_layout.itemAt(j).widget()
                    if isinstance(item, QWidget):
                        if isinstance(item, CollapsibleBox):
                            content_widget = item.content_area.widget()
                            if content_widget and content_widget.layout():
                                content_layout = content_widget.layout()
                                for k in range(content_layout.rowCount()):
                                    label_item = content_layout.itemAtPosition(k, 0)
                                    input_item = content_layout.itemAtPosition(k, 1)
                                    if label_item and input_item:
                                        key = label_item.widget().text()
                                        value = self.get_widget_value(input_item.widget())
                                        self.config[section][key] = value
                        else:
                            item_layout = item.layout()
                            if item_layout and item_layout.count() == 2:
                                key = item_layout.itemAt(0).widget().text()
                                value = self.get_widget_value(item_layout.itemAt(1).widget())
                                self.config[section][key] = value
        self.config = ConfigObj(self.config, configspec=self.spec_file)
        validator = Validator()
        self.config.validate(validator, preserve_errors=True)

    def update_gui_from_config(self):
        self._sync_alignment_widgets_from_config()
        for i in range(self.layout.count()):
            section_frame = self.layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text().strip('<b>').strip('</b>')
                section = helper.section_key_from_label(section)

                self.update_section_widgets(section_layout, self.config[section])

    def _sync_alignment_widgets_from_config(self):
        """Push the ALIGNMENT config values into the hand-rendered
        widgets (they live outside the auto-form scroll layout)."""
        align = self.config.get('ALIGNMENT', {})
        preset = str(align.get('bench sim preset', '')).strip()
        index = self.preset_combo.findText(preset)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self.probe_amp_field.setText(str(align.get('probe amplitude', '')))

    def update_section_widgets(self, section_layout, config_section):
        for i in range(1, section_layout.count()):
            item = section_layout.itemAt(i).widget()
            if isinstance(item, QWidget):
                if isinstance(item, CollapsibleBox):
                    try:
                        content_widget = item.content_area.widget()
                        if content_widget:
                            content_layout = content_widget.layout()
                            if content_layout:
                                for j in range(content_layout.rowCount()):
                                    label_item = content_layout.itemAtPosition(j, 0)
                                    input_item = content_layout.itemAtPosition(j, 1)
                                    if label_item and input_item:
                                        label = label_item.widget()
                                        input_widget = input_item.widget()
                                        key = label.text()
                                        if key in config_section:
                                            self.set_widget_value(input_widget, config_section[key])
                            else:
                                print("Warning: CollapsibleBox content widget has no layout")
                        else:
                            print("Warning: CollapsibleBox content area has no widget")
                    except Exception as e:
                        print(f"Error updating CollapsibleBox: {e}")
                else:
                    item_layout = item.layout()
                    if item_layout and item_layout.count() == 2:
                        label = item_layout.itemAt(0).widget()
                        input_widget = item_layout.itemAt(1).widget()
                        key = label.text()
                        if key in config_section:
                            self.set_widget_value(input_widget, config_section[key])

    def get_widget_value(self, widget):
        """Get the value from a widget"""
        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, QComboBox):
            # Dropdowns with a display-label map store the internal token
            # as itemData; plain combos use the visible text.
            if getattr(widget, '_uses_item_data', False):
                return widget.currentData()
            return widget.currentText()
        elif hasattr(widget, 'text_field') and isinstance(widget.text_field, QLineEdit):
            return widget.text_field.text()
        else:
            return str(widget.text())

    def set_widget_value(self, widget, value):
        """Set the value of an input widget"""
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QComboBox):
            if getattr(widget, '_uses_item_data', False):
                index = widget.findData(str(value).strip())
            else:
                index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)
            elif not getattr(widget, '_uses_item_data', False):
                widget.setCurrentText(str(value))
        elif hasattr(widget, 'text_field') and isinstance(widget.text_field, QLineEdit):
            widget.text_field.setText(str(value))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TokyoDriftConfigGUI()
    ex.show()
    sys.exit(app.exec_())
