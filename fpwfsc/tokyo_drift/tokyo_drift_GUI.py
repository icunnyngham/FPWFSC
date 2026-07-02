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
import sys
import threading
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton,
                             QScrollArea, QFrame, QToolButton, QSizePolicy,
                             QFileDialog, QGridLayout)
from PyQt5.QtCore import (Qt, pyqtSlot, QParallelAnimationGroup,
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
from fpwfsc.tokyo_drift.run import run

# Config sections owned by the top dropdowns rather than the form.
SELECTOR_SECTIONS = ("MODE",)


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

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tokyo Drift')
        self.resize(425, 875)

        main_layout = QVBoxLayout(self)

        # --- Top selectors: Hardware / Mode / Calibration --------------
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

        selector_layout.addWidget(QLabel("Calibration"), 2, 0)
        self.calibration_select = QComboBox()
        self.calibration_select.setFixedHeight(20)
        selector_layout.addWidget(self.calibration_select, 2, 1)

        main_layout.addLayout(selector_layout)

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

    def populate_calibration_selector(self):
        """Calibration choices are filtered to the selected mode."""
        mode = self.mode_select.currentText()
        profiles = ['None'] + helper.list_calibrations(mode_name=mode or None)

        current = str(self.config['MODE']['calibration profile'])
        self.calibration_select.clear()
        self.calibration_select.addItems(profiles)
        index = self.calibration_select.findText(current)
        if index >= 0:
            self.calibration_select.setCurrentIndex(index)

    def on_mode_changed(self, _mode_name):
        self.populate_calibration_selector()

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

        self.algorithm_thread = AlgorithmThread(
            camera=self.camera,
            aosystem=self.aosystem,
            config=self.config,
            spec_file=self.spec_file,
            my_event=self.my_event,
            plotter=self.plotter
        )
        self.algorithm_thread.start()

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
            if section in SELECTOR_SECTIONS:
                continue  # owned by the top dropdowns
            section_frame = QFrame()
            section_frame.setFrameShape(QFrame.StyledPanel)
            section_layout = QVBoxLayout(section_frame)
            section_layout.setSpacing(4)

            section_label = QLabel(f"<b>{section}</b>")
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

        spec = self.get_spec_for_key(f"{section}.{key}")

        if spec:
            if 'option(' in spec:
                input_widget = QComboBox()
                options = spec.split('option(')[1].split(')')[0].replace("'", "").split(',')
                input_widget.addItems([opt.strip() for opt in options])
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
        # Selector-owned section first
        self.config['MODE']['mode name'] = self.mode_select.currentText()
        self.config['MODE']['calibration profile'] = self.calibration_select.currentText()

        for i in range(self.layout.count()):
            section_frame = self.layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text().strip('<b>').strip('</b>')

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
        for i in range(self.layout.count()):
            section_frame = self.layout.itemAt(i).widget()
            if isinstance(section_frame, QFrame):
                section_layout = section_frame.layout()
                section_label = section_layout.itemAt(0).widget()
                section = section_label.text().strip('<b>').strip('</b>')

                self.update_section_widgets(section_layout, self.config[section])

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
            index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)
            else:
                widget.setCurrentText(str(value))
        elif hasattr(widget, 'text_field') and isinstance(widget.text_field, QLineEdit):
            widget.text_field.setText(str(value))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TokyoDriftConfigGUI()
    ex.show()
    sys.exit(app.exec_())
