"""Player bar under the 3D view: transport buttons, frame counter, seek bar, time, speed."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

SPEEDS = (("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0), ("Max", 0.0))


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of True values."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist(),
                    strict=True))


def format_time(seconds: float) -> str:
    minutes, s = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes):02d}:{s:04.1f}"


class SeekBar(QtWidgets.QWidget):
    """Video-style seek bar: loaded frames, played part and playhead; click or drag to seek."""

    seek = QtCore.Signal(int)
    MARGIN = 9

    def __init__(self, parent=None):
        super().__init__(parent)
        self.count = 0
        self.index = 0
        self.cached = np.zeros(0, bool)
        self._dragging = False
        self.setMinimumSize(120, 24)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click or drag to seek. Light gray: frames in memory")

    def set_count(self, n: int) -> None:
        self.count = n
        self.index = 0
        self.cached = np.zeros(n, bool)
        self.update()

    def set_index(self, i: int) -> None:
        if i != self.index:
            self.index = i
            self.update()

    def set_cached(self, mask: np.ndarray) -> None:
        self.cached = mask
        self.update()

    def _track(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.MARGIN, self.height() / 2 - 3,
                             max(self.width() - 2 * self.MARGIN, 1), 6)

    def _x(self, i: float) -> float:
        t = self._track()
        return t.left() + t.width() * i / max(self.count - 1, 1)

    def _index_at(self, x: float) -> int:
        t = self._track()
        u = min(max((x - t.left()) / t.width(), 0.0), 1.0)
        return int(round(u * max(self.count - 1, 0)))

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        t = self._track()
        p.setBrush(QtGui.QColor("#3a3a3e"))
        p.drawRoundedRect(t, 3, 3)
        if self.count > 0:
            half = t.width() / max(self.count - 1, 1) / 2
            p.setBrush(QtGui.QColor("#6b6f7a"))
            for a, b in true_runs(self.cached):
                x0, x1 = max(self._x(a) - half, t.left()), min(self._x(b) + half, t.right())
                p.drawRect(QtCore.QRectF(x0, t.top(), x1 - x0, t.height()))
            x = self._x(self.index)
            p.setBrush(QtGui.QColor(theme.ACCENT))
            p.drawRoundedRect(QtCore.QRectF(t.left(), t.top(), x - t.left(), t.height()), 3, 3)
            p.setBrush(QtGui.QColor("#ffffff"))
            p.drawEllipse(QtCore.QPointF(x, t.center().y()), 6.5, 6.5)
        p.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.count:
            self._dragging = True
            self._seek_to(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self._dragging:
            self._seek_to(x)
        elif self.count:
            QtWidgets.QToolTip.showText(event.globalPosition().toPoint(),
                                        f"frame {self._index_at(x)}", self)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    def wheelEvent(self, event) -> None:
        if self.count:
            step = -1 if event.angleDelta().y() > 0 else 1
            self.seek.emit(min(max(self.index + step, 0), self.count - 1))

    def _seek_to(self, x: float) -> None:
        i = self._index_at(x)
        if i != self.index:
            self.seek.emit(i)


class PlayerBar(QtWidgets.QWidget):
    """Transport buttons (actions owned by the main window), frame box, seek bar, speed."""

    seek = QtCore.Signal(int)
    speedChanged = QtCore.Signal(float)

    def __init__(self, actions: dict[str, QtGui.QAction], parent=None):
        super().__init__(parent)
        self.rate = 10.0
        self.count = 0
        self._updating = False
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 8, 3)
        layout.setSpacing(4)
        for name in ("first", "prev", "play", "next", "last"):
            layout.addWidget(self._button(actions[name]))
        layout.addSpacing(6)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.setFixedWidth(72)
        self.frame_spin.setToolTip("Frame number")
        self.frame_spin.valueChanged.connect(self._spin_changed)
        layout.addWidget(self.frame_spin)
        self.count_label = QtWidgets.QLabel("/ 0")
        layout.addWidget(self.count_label)
        layout.addSpacing(6)
        self.seekbar = SeekBar()
        self.seekbar.seek.connect(self.seek)
        layout.addWidget(self.seekbar, 1)
        self.time_label = QtWidgets.QLabel()
        self.time_label.setFont(QtGui.QFont("Consolas", 10))
        self.time_label.setToolTip("Time since the first frame / length of the test run")
        layout.addWidget(self.time_label)
        self.speed = QtWidgets.QComboBox()
        for label, value in SPEEDS:
            self.speed.addItem(label, value)
        self.speed.setCurrentIndex(2)
        self.speed.setToolTip("Playback speed (1× = recorded rate, Max = as fast as frames load)")
        self.speed.currentIndexChanged.connect(
            lambda: self.speedChanged.emit(self.speed.currentData()))
        layout.addWidget(self.speed)
        layout.addWidget(self._button(actions["loop"]))
        self.set_count(0, 10.0)

    @staticmethod
    def _button(action: QtGui.QAction) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setDefaultAction(action)
        button.setAutoRaise(True)
        button.setIconSize(QtCore.QSize(20, 20))
        return button

    def set_count(self, n: int, rate: float) -> None:
        self.count, self.rate = n, rate
        self._updating = True
        self.frame_spin.setRange(0, max(n - 1, 0))
        self._updating = False
        self.count_label.setText(f"/ {max(n - 1, 0)}")
        self.seekbar.set_count(n)
        self.set_index(0)

    def set_index(self, i: int) -> None:
        self._updating = True
        self.frame_spin.setValue(i)
        self._updating = False
        self.seekbar.set_index(i)
        total = max(self.count - 1, 0) / self.rate
        self.time_label.setText(f"{format_time(i / self.rate)} / {format_time(total)}")

    def set_speed(self, speed: float) -> None:
        k = self.speed.findData(speed)
        if k >= 0:
            self.speed.setCurrentIndex(k)

    def _spin_changed(self, value: int) -> None:
        if not self._updating:
            self.seek.emit(value)
