"""Playback clock and the layers drawn on top of the point cloud."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore


@dataclass
class Layer:
    """Points drawn over the cloud: a per-frame highlight (``func``) or fixed ``points``."""

    name: str
    color: str
    size: float
    func: Callable | None = None  # frame -> boolean mask / Frame / (N, 3) points, every frame
    points: np.ndarray | None = None
    visible: bool = True
    count: int = 0
    error: Exception | None = None


class Rate:
    """Events per second over the last second."""

    def __init__(self):
        self._times: deque[float] = deque(maxlen=500)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def value(self) -> float:
        now = time.perf_counter()
        while self._times and now - self._times[0] > 1.0:
            self._times.popleft()
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)


class Player(QtCore.QObject):
    """Advances frames at the recording's rate times ``speed`` (0 = as fast as frames load).

    The playhead only moves to frames that are already loaded, so slow loading shows up as
    a lower frame rate (``stalls`` counts the waits) instead of skipped frames.
    """

    stateChanged = QtCore.Signal(bool)

    def __init__(self, window):
        super().__init__(window)
        self.w = window
        self.speed = 1.0
        self.loop = True
        self.stalls = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def play(self) -> None:
        w = self.w
        if w.rec is None or self.playing:
            return
        if w.index >= len(w.rec) - 1:
            w.show_frame(0)
        self._apply_interval()
        self.timer.start()
        self.stateChanged.emit(True)
        w.prefetch()

    def pause(self) -> None:
        if self.playing:
            self.timer.stop()
            self.stateChanged.emit(False)
            self.w.prefetch()

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def set_speed(self, speed: float) -> None:
        self.speed = speed
        if self.playing:
            self._apply_interval()
            self.w.prefetch()

    def _apply_interval(self) -> None:
        rate = self.w.rec.rate_hz if self.w.rec is not None else 10.0
        interval = 0 if self.speed <= 0 else max(1, round(1000.0 / (rate * self.speed)))
        self.timer.setInterval(interval)

    def _tick(self) -> None:
        w = self.w
        if w.rec is None:
            self.pause()
            return
        nxt = w.index + 1
        if nxt >= len(w.rec):
            if not self.loop:
                self.pause()
                return
            nxt = 0
        if w.rec.is_cached(nxt):
            w.show_frame(nxt)
        else:
            self.stalls += 1
            w.prefetch(nxt)
