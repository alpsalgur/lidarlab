"""Inspector: the distribution of one field in the current frame, as loaded and after the
pipeline, optionally only inside the box selection."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .display import CATEGORICAL, SOLID, color_values

LOADED_COLOR = "#5a5f6a"
BINS = 60


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


class HistogramWidget(QtWidgets.QWidget):
    """Bars for the frame as loaded (gray) with the pipeline's output on top (blue): what the
    pipeline removed shows as gray. Hover a bar for its counts."""

    MARGINS = (52, 22, 10, 26)  # left, top, right, bottom

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.field = ""
        self.labels: list[str] | None = None  # categories, or None for numeric bins
        self.edges = np.zeros(0)
        self.loaded = np.zeros(0)
        self.after = np.zeros(0)
        self.log = False
        self.show_after = True

    def set_data(self, field: str, loaded, after, edges=None, labels=None, log=False,
                 show_after=True) -> None:
        self.field, self.loaded, self.after = field, np.asarray(loaded), np.asarray(after)
        self.edges = np.zeros(0) if edges is None else np.asarray(edges)
        self.labels, self.log, self.show_after = labels, log, show_after
        self.update()

    def _plot(self) -> QtCore.QRectF:
        left, top, right, bottom = self.MARGINS
        return QtCore.QRectF(left, top, max(self.width() - left - right, 10),
                             max(self.height() - top - bottom, 10))

    def _scale(self, value: float, peak: float) -> float:
        if self.log:
            return np.log10(value + 1) / max(np.log10(peak + 1), 1e-9)
        return value / max(peak, 1)

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(theme.BASE))
        n = len(self.loaded)
        plot = self._plot()
        p.setPen(QtGui.QColor(theme.BORDER))
        p.drawLine(plot.bottomLeft(), plot.bottomRight())
        if not n:
            p.setPen(QtGui.QColor(theme.MUTED))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "no data")
            p.end()
            return
        peak = float(max(self.loaded.max(), self.after.max() if self.after.size else 0, 1))
        width = plot.width() / n
        gap = width * 0.18 if self.labels is not None else 0.0
        series = [(self.loaded, LOADED_COLOR)]
        if self.show_after:
            series.append((self.after, theme.ACCENT))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        for counts, color in series:
            p.setBrush(QtGui.QColor(color))
            for k, value in enumerate(counts):
                if value <= 0:
                    continue
                h = self._scale(float(value), peak) * plot.height()
                p.drawRect(QtCore.QRectF(plot.left() + k * width + gap / 2, plot.bottom() - h,
                                         max(width - gap, 1.0), h))
        p.setPen(QtGui.QColor(theme.MUTED))
        font = p.font()
        font.setPointSizeF(8.0)
        p.setFont(font)
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        p.drawText(QtCore.QRectF(0, plot.top() - 7, plot.left() - 6, 14), right, _fmt(peak))
        p.drawText(QtCore.QRectF(0, plot.bottom() - 7, plot.left() - 6, 14), right, "0")
        if self.log:
            p.drawText(QtCore.QRectF(0, plot.center().y() - 7, plot.left() - 6, 14), right, "log")
        center = QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
        if self.labels is not None:
            metrics = QtGui.QFontMetrics(font)
            for k, label in enumerate(self.labels):
                text = metrics.elidedText(label, QtCore.Qt.TextElideMode.ElideRight, int(width))
                p.drawText(QtCore.QRectF(plot.left() + k * width, plot.bottom() + 4, width, 16),
                           center, text)
        elif self.edges.size:
            for k in range(5):
                u = k / 4
                value = self.edges[0] + u * (self.edges[-1] - self.edges[0])
                x = plot.left() + u * plot.width()
                p.drawText(QtCore.QRectF(x - 40, plot.bottom() + 4, 80, 16), center, _fmt(value))
        top = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        x = plot.left()
        legend = [("loaded", LOADED_COLOR), ("after pipeline", theme.ACCENT)][:len(series)]
        for label, color in legend:
            p.fillRect(QtCore.QRectF(x, 6, 9, 9), QtGui.QColor(color))
            p.drawText(QtCore.QRectF(x + 13, 2, 110, 16), top, label)
            x += 100
        p.end()

    def mouseMoveEvent(self, event) -> None:
        n = len(self.loaded)
        plot = self._plot()
        x = event.position().x()
        if not n or not plot.left() <= x <= plot.right():
            QtWidgets.QToolTip.hideText()
            return
        k = min(int((x - plot.left()) / plot.width() * n), n - 1)
        if self.labels is not None:
            what = self.labels[k]
        else:
            what = f"{self.field} {_fmt(self.edges[k])} – {_fmt(self.edges[k + 1])}"
        text = f"{what}\nloaded: {int(self.loaded[k]):,}"
        if self.show_after:
            text += f"\nafter pipeline: {int(self.after[k]):,}"
        QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), text, self)


class InspectorPanel(QtWidgets.QWidget):
    """Field histogram and statistics of the current frame."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.w = window
        self.field = QtWidgets.QComboBox()
        self.field.setToolTip("Field to show")
        self.log = QtWidgets.QCheckBox("Log")
        self.log.setToolTip("Logarithmic counts: makes small groups (noise) visible")
        self.selection_only = QtWidgets.QCheckBox("Selection")
        self.selection_only.setToolTip("Only the points inside the box selection (S in the view)")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.field, 1)
        row.addWidget(self.log)
        row.addWidget(self.selection_only)
        self.histogram = HistogramWidget()
        self.stats = QtWidgets.QLabel()
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stats.setStyleSheet(f"color: {theme.MUTED};")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(row)
        layout.addWidget(self.histogram, 1)
        layout.addWidget(self.stats)
        self.field.currentTextChanged.connect(lambda _: self.refresh())
        self.log.toggled.connect(lambda _: self.refresh())
        self.selection_only.toggled.connect(lambda _: self.refresh())

    def set_fields(self, modes: list[str]) -> None:
        current = self.field.currentText()
        with QtCore.QSignalBlocker(self.field):
            self.field.clear()
            self.field.addItems([m for m in modes if m != SOLID])
            if current and self.field.findText(current) >= 0:
                self.field.setCurrentText(current)
            elif self.field.findText("intensity") >= 0:
                self.field.setCurrentText("intensity")

    def refresh(self) -> None:
        w = self.w
        if w.raw is None or w.output is None or not self.isVisible():
            return
        name = self.field.currentText()
        raw, out = w.raw, w.output
        if self.selection_only.isChecked() and w.selection is not None:
            raw, out = raw[w.selection.mask(raw.xyz)], out[w.selection.mask(out.xyz)]
        loaded, after = color_values(raw, name), color_values(out, name)
        if loaded is None:
            self.histogram.set_data(name, [], [])
            self.stats.setText(f"{name}: not in this frame")
            return
        after = np.zeros(0, loaded.dtype) if after is None else after
        show_after = w.pipeline.active
        if name in CATEGORICAL:
            classes = np.union1d(np.unique(loaded), np.unique(after)).astype(int)
            names = w.rec.info.get("labels", {}) if w.rec is not None else {}
            labels = [f"echo {c}" if name == "echo" else names.get(str(c), str(c))
                      for c in classes]
            def counts(values):
                return np.bincount(np.searchsorted(classes, values), minlength=len(classes))

            c_loaded, c_after = counts(loaded), counts(after)
            self.histogram.set_data(name, c_loaded, c_after, labels=labels,
                                    log=self.log.isChecked(), show_after=show_after)
            total = max(len(after) if show_after else len(loaded), 1)
            shown = c_after if show_after else c_loaded
            self.stats.setText("   ".join(f"{label}: {100 * c / total:.1f} %"
                                          for label, c in zip(labels, shown, strict=True)))
            return
        loaded = loaded[np.isfinite(loaded)]
        after = after[np.isfinite(after)]
        if not loaded.size:
            self.histogram.set_data(name, [], [])
            self.stats.setText(f"{name}: no finite values")
            return
        lo, hi = float(loaded.min()), float(loaded.max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1.0, BINS + 1)
        c_loaded, _ = np.histogram(loaded, edges)
        c_after, _ = np.histogram(after, edges)
        self.histogram.set_data(name, c_loaded, c_after, edges=edges, log=self.log.isChecked(),
                                show_after=show_after)
        v = after if show_after and after.size else loaded
        p1, median, p99 = np.percentile(v, [1, 50, 99])
        self.stats.setText(f"{'after pipeline' if v is after else 'loaded'}: {len(v):,} points · "
                           f"mean {_fmt(v.mean())} · median {_fmt(median)} · std "
                           f"{_fmt(v.std())} · 1–99 % {_fmt(p1)} – {_fmt(p99)}")
