"""PyQtGraph live plotter for the tokyo_drift pipeline.

Layout:
- Top row, the alignment view: Source (detector frame with the current
  calibration applied), Ideal (model-native simulated PSF), and their
  Difference. Source/Ideal are drawn with the inferno colormap on a log
  scale; the Difference is drawn linearly with a symmetric
  blue-white-red map.
- Bottom row: Strehl-ratio history and the current mode-coefficient
  vector.

Thread-safety follows the FPWFSC plotter convention: ``update()`` may
be called from the algorithm thread; it copies the arrays and emits a
signal so the actual drawing happens in the GUI thread.
"""
import sys

import numpy as np
import matplotlib
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QObject
import pyqtgraph as pg

# Displayed dynamic range of log-scaled PSF panels, in decades below
# the peak (matches the bench notebooks' LogNorm(vmin=1e-4, vmax=1)).
LOG_DECADES = 4.0


def inferno_lut(nsteps=256):
    """Inferno colormap as a pyqtgraph lookup table."""
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps["inferno"]
    else:  # matplotlib < 3.5
        cmap = matplotlib.cm.get_cmap("inferno")
    return (cmap(np.linspace(0, 1, nsteps))[:, :3] * 255).astype(np.uint8)


def bwr_lut(nsteps=256):
    """Symmetric blue-white-red lookup table for difference images."""
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps["bwr"]
    else:
        cmap = matplotlib.cm.get_cmap("bwr")
    return (cmap(np.linspace(0, 1, nsteps))[:, :3] * 255).astype(np.uint8)


def log_display(image, vmin_decades=LOG_DECADES):
    """Peak-normalized log10 image for LogNorm-style display.

    Returns ``log10(|image| / max)`` clipped to ``[-vmin_decades, 0]``
    — physical units (decades below peak), so a colorbar over these
    values is directly readable.
    """
    image = np.asarray(image, dtype=float)
    mx = np.max(np.abs(image))
    if mx <= 0:
        return np.full_like(image, -vmin_decades)
    logim = np.log10(np.abs(image) / mx + 10.0 ** (-vmin_decades - 2))
    return np.clip(logim, -vmin_decades, 0.0)


def _pg_colormap(lut):
    """Wrap a (N, 3) uint8 LUT as a pyqtgraph ColorMap."""
    return pg.ColorMap(np.linspace(0.0, 1.0, len(lut)), lut)


class PlotterSignals(QObject):
    """Thread-safe channel from the algorithm thread to the GUI thread."""
    update_signal = pyqtSignal(dict)


class LivePlotter(QtWidgets.QWidget):
    """Live view of a tokyo_drift run.

    ``update(payload)`` accepts a dict; all keys optional:

    - ``n_iter``: total planned iterations
    - ``source``: 2-D detector frame with current calibration applied
    - ``ideal``: 2-D model-native simulated PSF
    - ``strehls``: 1-D Strehl history (NaN for not-yet-run iterations)
    - ``mode_coeffs``: 1-D current mode-coefficient vector
    """

    def __init__(self, figsize=(900, 650)):
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication(sys.argv)
        else:
            self.app = QtWidgets.QApplication.instance()

        super().__init__()

        self.signals = PlotterSignals()
        self.signals.update_signal.connect(self._update_plots)

        self.setWindowTitle("Tokyo Drift - Live Plotter")
        self.resize(figsize[0], figsize[1])

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self._psf_cmap = _pg_colormap(inferno_lut())
        self._diff_cmap = _pg_colormap(bwr_lut())

        # --- Top row: alignment view (source / ideal / difference) ----
        # Source/Ideal display log10(rel. intensity) over LOG_DECADES
        # decades; Difference displays raw fraction-of-peak residuals on
        # a symmetric scale. Every panel carries a labeled colorbar.
        self.image_plots = {}
        self.image_items = {}
        self.color_bars = {}
        self._raw_images = {}
        for col, (key, title) in enumerate(
                [("source", "Source (calibrated)"),
                 ("ideal", "Ideal"),
                 ("diff", "Difference")]):
            plot = pg.PlotWidget()
            plot.setTitle(title)
            plot.setAspectLocked(True)
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            img = pg.ImageItem()
            plot.addItem(img)
            if key == "diff":
                bar = pg.ColorBarItem(colorMap=self._diff_cmap,
                                      values=(-1e-3, 1e-3), width=15,
                                      label="fraction of peak")
            else:
                bar = pg.ColorBarItem(colorMap=self._psf_cmap,
                                      values=(-LOG_DECADES, 0.0), width=15,
                                      label="log10 relative intensity")
            bar.setImageItem(img, insert_in=plot.getPlotItem())
            layout.addWidget(plot, 0, col)
            self.image_plots[key] = plot
            self.image_items[key] = img
            self.color_bars[key] = bar
            self._raw_images[key] = None

        # --- Bottom row: Strehl history + mode coefficients -----------
        self.strehl_plot = pg.PlotWidget()
        self.strehl_plot.setTitle("Strehl Ratio")
        self.strehl_plot.setLabel('left', 'Strehl')
        self.strehl_plot.setLabel('bottom', 'Iteration')
        self.strehl_plot.showGrid(x=True, y=True)
        self.strehl_plot.setYRange(0, 1.1)
        self.strehl_curve = pg.PlotDataItem(pen=pg.mkPen('g', width=2))
        self.strehl_plot.addItem(self.strehl_curve)
        layout.addWidget(self.strehl_plot, 1, 0)

        self.coeff_plot = pg.PlotWidget()
        self.coeff_plot.setTitle("Mode Coefficients")
        self.coeff_plot.setLabel('left', 'Amplitude')
        self.coeff_plot.setLabel('bottom', 'Mode index')
        self.coeff_plot.showGrid(x=True, y=True)
        self.coeff_bars = None
        # Calibration sweep curves (rotation / scale) borrow this panel
        self.sweep_curve = pg.PlotDataItem(pen=pg.mkPen('c', width=2))
        self.coeff_plot.addItem(self.sweep_curve)
        layout.addWidget(self.coeff_plot, 1, 1, 1, 2)

        # Stage-annotation items on the history axes (calibration mode)
        self._stage_marks = []

        # Pixel-value readout on hover over any image panel
        self.pixel_label = QtWidgets.QLabel("")
        self.pixel_label.setStyleSheet(
            "color: white; background: rgba(0,0,0,150); padding: 2px;")
        layout.addWidget(self.pixel_label, 2, 0, 1, 3)
        self._proxies = [
            pg.SignalProxy(self.image_plots[key].scene().sigMouseMoved,
                           rateLimit=30,
                           slot=lambda evt, k=key: self._on_mouse(evt, k))
            for key in self.image_plots
        ]

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_events)
        self.timer.start(50)

        self.closed = False
        self.show()

    @staticmethod
    def _ensure_2d(data):
        """Coerce input (HCIPy field, 1-D, complex, ...) to a real 2-D array."""
        if data is None:
            return np.zeros((1, 1))
        if hasattr(data, 'shaped'):
            data = data.shaped
        data = np.asarray(data)
        if np.iscomplexobj(data):
            data = np.abs(data)
        if data.ndim == 1:
            n = int(np.sqrt(data.size))
            data = data[:n * n].reshape(n, n)
        elif data.ndim > 2:
            data = data[..., 0]
        return data.astype(float)

    def _clear_stage_marks(self):
        for item in self._stage_marks:
            self.strehl_plot.removeItem(item)
        self._stage_marks = []

    def _on_mouse(self, evt, key):
        raw = self._raw_images.get(key)
        if raw is None:
            return
        pos = evt[0]
        pt = self.image_plots[key].plotItem.vb.mapSceneToView(pos)
        col, row = int(round(pt.x())), int(round(pt.y()))
        h, w = raw.shape[:2]
        if 0 <= row < h and 0 <= col < w:
            self.pixel_label.setText(
                f"{key}  x={col}  y={row}  val={raw[row, col]:.4g}")
        else:
            self.pixel_label.setText("")

    def closeEvent(self, event):
        self.closed = True
        self.timer.stop()
        event.accept()

    def close(self):
        self.closed = True
        self.timer.stop()
        super().close()

    def process_events(self):
        if not self.closed:
            QtWidgets.QApplication.processEvents()

    def update(self, payload):
        """Thread-safe update; may be called from the algorithm thread."""
        if self.closed:
            return
        copied = {}
        for key, value in payload.items():
            copied[key] = np.array(value) if isinstance(
                value, (np.ndarray, list, tuple)) else value
        self.signals.update_signal.emit(copied)

    @pyqtSlot(dict)
    def _update_plots(self, payload):
        if self.closed:
            return
        try:
            source = payload.get("source")
            ideal = payload.get("ideal")

            if source is not None:
                source = self._ensure_2d(source).T
                self._raw_images["source"] = source
                self.image_items["source"].setImage(log_display(source),
                                                    autoLevels=False)
                self.color_bars["source"].setLevels((-LOG_DECADES, 0.0))
            if ideal is not None:
                ideal = self._ensure_2d(ideal).T
                self._raw_images["ideal"] = ideal
                self.image_items["ideal"].setImage(log_display(ideal),
                                                   autoLevels=False)
                self.color_bars["ideal"].setLevels((-LOG_DECADES, 0.0))

            # Difference of the peak-normalized linear images, displayed
            # in physical units (fraction of peak) on a symmetric scale
            # — the colorbar tells you how big the residual really is.
            src, ide = self._raw_images["source"], self._raw_images["ideal"]
            if (src is not None and ide is not None
                    and src.shape == ide.shape
                    and src.max() > 0 and ide.max() > 0):
                diff = src / src.max() - ide / ide.max()
                self._raw_images["diff"] = diff
                span = float(np.max(np.abs(diff))) or 1e-12
                self.image_items["diff"].setImage(diff, autoLevels=False)
                self.color_bars["diff"].setLevels((-span, span))

            source_title = payload.get("source_title")
            if source_title:
                self.image_plots["source"].setTitle(source_title)
            ideal_title = payload.get("ideal_title")
            if ideal_title:
                self.image_plots["ideal"].setTitle(ideal_title)

            # Calibration sweep curve (rotation / scale) borrows the
            # mode-coefficients panel; the next mode_coeffs payload
            # reclaims it.
            curve = payload.get("curve")
            if curve is not None:
                if self.coeff_bars is not None:
                    self.coeff_plot.removeItem(self.coeff_bars)
                    self.coeff_bars = None
                x, y = curve
                self.sweep_curve.setData(np.asarray(x, dtype=float),
                                         np.asarray(y, dtype=float))
                self.coeff_plot.setTitle(payload.get("curve_title", "Score"))
                self.coeff_plot.setLabel('left', 'score')
                self.coeff_plot.setLabel('bottom', 'swept value')
                self.coeff_plot.enableAutoRange()

            # Calibration progress (RMS of the diff image per stage)
            # borrows the Strehl axes, with dashed stage annotations;
            # the next strehls payload reclaims them.
            progress = payload.get("progress")
            if progress is not None:
                self._clear_stage_marks()
                x = np.asarray(progress.get("x", []), dtype=float)
                y = np.asarray(progress.get("y", []), dtype=float)
                self.strehl_curve.setData(x, y, symbol='o')
                self.strehl_plot.setTitle(
                    progress.get("title", "Calibration progress"))
                self.strehl_plot.setLabel(
                    'left', progress.get("ylabel", "RMS(diff)"))
                self.strehl_plot.setLabel('bottom', 'calibration step')
                self.strehl_plot.enableAutoRange()
                ymax = float(np.max(y)) if y.size else 1.0
                for mark_x, label in progress.get("stage_marks", []):
                    line = pg.InfiniteLine(
                        pos=float(mark_x), angle=90,
                        pen=pg.mkPen((180, 180, 180), style=Qt.DashLine))
                    self.strehl_plot.addItem(line)
                    text = pg.TextItem(str(label), color=(220, 220, 220),
                                       anchor=(0, 1))
                    text.setPos(float(mark_x), ymax)
                    self.strehl_plot.addItem(text)
                    self._stage_marks.extend([line, text])

            strehls = payload.get("strehls")
            n_iter = payload.get("n_iter")
            if strehls is not None:
                strehls = np.asarray(strehls, dtype=float)
                valid = ~np.isnan(strehls)
                iterations = np.arange(strehls.size)[valid]
                self._clear_stage_marks()
                self.strehl_plot.setLabel('left', 'Strehl')
                self.strehl_plot.setLabel('bottom', 'Iteration')
                self.strehl_plot.setYRange(0, 1.1)
                if n_iter:
                    self.strehl_plot.setXRange(0, n_iter - 1)
                if iterations.size:
                    self.strehl_curve.setData(iterations, strehls[valid],
                                              symbol=None)
                    self.strehl_plot.setTitle(
                        f"Strehl Ratio = {strehls[valid][-1]:.3f}")

            coeffs = payload.get("mode_coeffs")
            if coeffs is not None:
                coeffs = np.asarray(coeffs, dtype=float).ravel()
                self.sweep_curve.setData([], [])
                self.coeff_plot.setTitle("Mode Coefficients")
                self.coeff_plot.setLabel('left', 'Amplitude')
                self.coeff_plot.setLabel('bottom', 'Mode index')
                if self.coeff_bars is not None:
                    self.coeff_plot.removeItem(self.coeff_bars)
                self.coeff_bars = pg.BarGraphItem(
                    x=np.arange(coeffs.size), height=coeffs, width=0.8,
                    brush=pg.mkBrush(200, 120, 40))
                self.coeff_plot.addItem(self.coeff_bars)

            iteration = payload.get("iteration")
            if iteration is not None and n_iter:
                self.image_plots["source"].setTitle(
                    f"Source (calibrated) - iter {iteration}/{n_iter}")
        except Exception as exc:
            print(f"Error updating plots: {exc}")
            import traceback
            traceback.print_exc()

    def execute(self):
        """Run the Qt event loop (standalone use)."""
        if not self.closed and hasattr(self, 'app'):
            self.app.exec_()


if __name__ == "__main__":
    # Standalone demo with dummy data.
    import time

    plotter = LivePlotter()

    n_iter = 30
    yy, xx = np.mgrid[-64:64, -64:64]
    ideal_psf = np.exp(-(xx**2 + yy**2) / (2 * 4.0**2))
    strehls = np.full(n_iter, np.nan)

    rng = np.random.default_rng(0)
    for i in range(10):
        strehls[i] = 0.4 + 0.05 * i
        blur = 8.0 - 0.5 * i
        source = (np.exp(-((xx - 2)**2 + (yy + 3)**2) / (2 * blur**2))
                  + rng.normal(0, 0.002, ideal_psf.shape))
        plotter.update({
            "n_iter": n_iter,
            "iteration": i + 1,
            "source": source,
            "ideal": ideal_psf,
            "strehls": strehls,
            "mode_coeffs": rng.normal(0, 1.0 / (i + 1), 10),
        })
        time.sleep(0.5)

    plotter.execute()
