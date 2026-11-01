"""File > Create Sample Test Run: a synthetic multi-echo recording with rain, spray and
ground-truth labels, so a fresh installation has something to open."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from PySide6 import QtCore

from lidarlab.datasets import synthetic


class SampleJob(QtCore.QObject):
    """Generates the run in a thread; ``progress(done, total)`` and ``finished(path or
    exception)`` arrive in the GUI thread."""

    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(object)

    def __init__(self, out_dir, frames: int = 100, parent=None):
        super().__init__(parent)
        self.out_dir = Path(out_dir)
        self.frames = frames
        self._thread = threading.Thread(target=self._run, name="lidarlab-sample", daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def _run(self) -> None:
        workers = max(1, min(4, (os.cpu_count() or 2) // 2))
        try:
            synthetic.generate(self.out_dir, synthetic.Config(frames=self.frames),
                               workers=workers, log=self._log)
        except Exception as e:  # noqa: BLE001 - shown by the window
            self.finished.emit(e)
            return
        self.finished.emit(self.out_dir)

    def _log(self, message: str) -> None:
        match = re.match(r"(\d+)/(\d+) frames", message)
        if match:
            self.progress.emit(int(match.group(1)), int(match.group(2)))


def free_folder(parent: Path, name: str) -> Path:
    """``parent/name``, or ``name_2``, ``name_3``… if it exists."""
    path, k = parent / name, 2
    while path.exists():
        path, k = parent / f"{name}_{k}", k + 1
    return path
