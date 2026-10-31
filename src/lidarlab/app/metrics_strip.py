"""Metric strip above the player bar: one value per frame across the whole test run, drawn in
line with the seek bar below it. Click to go to a frame."""

from __future__ import annotations

import threading

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from lidarlab.core.metrics import METRICS, MetricStore, compute_metrics

from . import theme

MARK_COLOR = "#ffd54f"


class MetricsJob(QtCore.QObject):
    """Measures all frames in a thread; ``progress(done, total)`` arrives in the GUI thread."""

    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(int)

    def __init__(self, rec, pipeline, store: MetricStore, noise_labels, select, parent=None):
        super().__init__(parent)
        self.cancel_requested = False
        self.store = store
        self._args = (rec, pipeline, store, None, None, noise_labels, select)
        self._thread = threading.Thread(target=self._run, name="lidarlab-metrics-job",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self.cancel_requested = True

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        done = compute_metrics(*self._args, progress=self._progress)
        self.finished.emit(done)

    def _progress(self, done: int, total: int) -> bool:
        self.progress.emit(done, total)
        return not self.cancel_requested


class MetricsStrip(QtWidgets.QWidget):
    """The chosen metric over all frames; the current frame and bookmarks are marked."""

    seek = QtCore.Signal(int)
    computeRequested = QtCore.Signal()

    def __init__(self, seekbar, parent=None):
        super().__init__(parent)
        self.seekbar = seekbar
        self.store: MetricStore | None = None
        self.index = 0
        self.marks: list[int] = []
        self.setFixedHeight(62)
        self.setMouseTracking(True)
        self.metric_box = QtWidgets.QComboBox(self)
        for key, (label, _unit) in METRICS.items():
            self.metric_box.addItem(label, key)
        self.metric_box.setCurrentIndex(self.metric_box.findData("removed_pct"))
        self.metric_box.setToolTip("Metric per frame")
        self.metric_box.currentIndexChanged.connect(self.update)
        self.compute = QtWidgets.QToolButton(self)
        self.compute.setText("Compute all")
        self.compute.setToolTip("Measure every frame in the background (runs the pipeline); "
                                "frames shown while playing are measured anyway")
        self.compute.clicked.connect(self.computeRequested)
        self.status = QtWidgets.QLabel(self)
        self.status.setStyleSheet(f"color: {theme.MUTED};")
        seekbar.installEventFilter(self)  # follow the seek bar when the player bar resizes

    @property
    def key(self) -> str:
        return self.metric_box.currentData()

    def set_store(self, store: MetricStore | None) -> None:
        self.store = store
        self.update()

    def set_index(self, i: int) -> None:
        if i != self.index:
            self.index = i
            self.update()

    def set_marks(self, frames: list[int]) -> None:
        self.marks = list(frames)
        self.update()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.seekbar and event.type() in (QtCore.QEvent.Type.Resize,
                                                    QtCore.QEvent.Type.Move):
            self._place_controls()
            self.update()
        return False

    def resizeEvent(self, event) -> None:
        self._place_controls()
        super().resizeEvent(event)

    def _plot(self) -> QtCore.QRectF:
        """Chart area, with the same horizontal extent as the seek bar's track."""
        track = self.seekbar._track()
        left = self.mapFromGlobal(self.seekbar.mapToGlobal(
            QtCore.QPoint(int(track.left()), 0))).x()
        right = self.mapFromGlobal(self.seekbar.mapToGlobal(
            QtCore.QPoint(int(track.right()), 0))).x()
        if right - left < 20:  # not laid out yet
            left, right = 220, max(self.width() - 220, 240)
        return QtCore.QRectF(left, 6, right - left, self.height() - 12)

    def _place_controls(self) -> None:
        left = int(self._plot().left()) - 12
        width = max(left - 8, 80)
        self.metric_box.setGeometry(8, 4, width, 24)
        self.compute.setGeometry(8, 32, 82, 24)
        self.status.setGeometry(96, 32, max(width - 88, 10), 24)

    def _x(self, plot: QtCore.QRectF, i: float) -> float:
        n = self.store.n if self.store else 1
        return plot.left() + plot.width() * i / max(n - 1, 1)

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        plot = self._plot()
        p.fillRect(plot, QtGui.QColor(theme.BASE))
        store = self.store
        label, unit = METRICS[self.key]
        values = store.values[self.key] if store is not None else np.zeros(0)
        known = np.isfinite(values)
        p.setPen(QtGui.QColor(theme.MUTED))
        if not known.any():
            p.drawText(plot, QtCore.Qt.AlignmentFlag.AlignCenter,
                       "no values yet: play, or Compute all")
        else:
            lo, hi = float(values[known].min()), float(values[known].max())
            if unit == "%" or self.key in ("points", "kept", "selected"):
                lo = min(lo, 0.0)
            if hi - lo < 1e-9:
                hi = lo + 1.0

            def y(v: float) -> float:
                return plot.bottom() - (v - lo) / (hi - lo) * (plot.height() - 4) - 2

            path, area = QtGui.QPainterPath(), QtGui.QPainterPath()
            run: list[tuple[float, float]] = []
            dots: list[tuple[float, float]] = []  # values with no measured neighbour

            def flush() -> None:
                if len(run) == 1:
                    dots.append(run.pop())
                if not run:
                    return
                path.moveTo(*run[0])
                area.moveTo(run[0][0], plot.bottom())
                for point in run:
                    path.lineTo(*point)
                    area.lineTo(*point)
                area.lineTo(run[-1][0], plot.bottom())
                area.closeSubpath()
                run.clear()

            for i, v in enumerate(values):
                if np.isfinite(v):
                    run.append((self._x(plot, i), y(float(v))))
                else:
                    flush()
            flush()
            fill = QtGui.QColor(theme.ACCENT)
            fill.setAlpha(60)
            p.fillPath(area, fill)
            p.setPen(QtGui.QPen(QtGui.QColor(theme.ACCENT), 1.4))
            p.drawPath(path)
            p.setBrush(QtGui.QColor(theme.ACCENT))
            for x, y_dot in dots:
                p.drawEllipse(QtCore.QPointF(x, y_dot), 2.2, 2.2)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.setPen(QtGui.QColor(theme.MUTED))
            font = p.font()
            font.setPointSizeF(8.0)
            p.setFont(font)
            p.drawText(QtCore.QRectF(plot.left() + 4, plot.top(), 120, 12),
                       QtCore.Qt.AlignmentFlag.AlignLeft, f"{hi:.4g}{unit}")
        for frame in self.marks:
            p.setPen(QtGui.QPen(QtGui.QColor(MARK_COLOR), 1.0))
            x = self._x(plot, frame)
            p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
        if store is not None and store.n:
            x = self._x(plot, self.index)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.2))
            p.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
            current = values[self.index] if self.index < len(values) else np.nan
            text = f"{current:,.4g}{unit}" if np.isfinite(current) else "–"
            p.setPen(QtGui.QColor(theme.TEXT))
            p.drawText(QtCore.QRectF(plot.right() + 10, plot.top(),
                                     self.width() - plot.right() - 14, plot.height()),
                       QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                       f"{text}\n{label}")
        p.end()

    def _frame_at(self, x: float) -> int | None:
        plot = self._plot()
        if self.store is None or not plot.left() - 3 <= x <= plot.right() + 3:
            return None
        u = min(max((x - plot.left()) / plot.width(), 0.0), 1.0)
        return int(round(u * max(self.store.n - 1, 0)))

    def mousePressEvent(self, event) -> None:
        i = self._frame_at(event.position().x())
        if i is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.seek.emit(i)

    def mouseMoveEvent(self, event) -> None:
        i = self._frame_at(event.position().x())
        if i is None:
            QtWidgets.QToolTip.hideText()
            return
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self.seek.emit(i)
        label, unit = METRICS[self.key]
        v = self.store.values[self.key][i]
        value = f"{v:,.4g}{unit}" if np.isfinite(v) else "not measured"
        QtWidgets.QToolTip.showText(event.globalPosition().toPoint(),
                                    f"frame {i}: {value}\n{label}", self)
