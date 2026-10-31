"""Export: run the current pipeline over the open test run and write the frames."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from lidarlab import export_run
from lidarlab.core.batch import default_workers

from . import theme
from .paths import data_dir

FORMAT_LABELS = {"npz": "npz, the source's layout (a test run you can open)",
                 "ply": "PLY, binary (CloudCompare, Open3D)",
                 "pcd": "PCD, binary (PCL)"}


class ExportJob(QtCore.QObject):
    """Runs ``export_run`` in a thread. ``progress(done, total)`` and ``finished(report or
    exception)`` arrive in the GUI thread."""

    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(object)

    def __init__(self, source, out_dir, pipeline, frames=None, fmt="npz", workers=None,
                 compress=True, overwrite=False, parent=None):
        super().__init__(parent)
        self.out_dir = Path(out_dir)
        self.fmt = fmt
        self.report = None
        self.error: Exception | None = None
        self.cancel_requested = False
        self._args = (source, out_dir, pipeline, frames, fmt, workers, False, compress, overwrite)
        self._thread = threading.Thread(target=self._run, name="lidarlab-export-job", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self.cancel_requested = True

    @property
    def done(self) -> bool:
        return not self._thread.is_alive() and (self.report is not None or self.error is not None)

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        try:
            self.report = export_run(*self._args, progress=self._progress)
        except Exception as e:  # noqa: BLE001 - shown by the window
            self.error = e
        self.finished.emit(self.report if self.error is None else self.error)

    def _progress(self, done: int, total: int, result) -> bool:
        self.progress.emit(done, total)
        return not self.cancel_requested


def default_output(run_name: str) -> Path:
    return data_dir() / "exports" / f"{run_name}_{datetime.now():%Y%m%d-%H%M%S}"


class ExportDialog(QtWidgets.QDialog):
    """Where to, which frames, which format; shows the steps that will run."""

    def __init__(self, window):
        super().__init__(window)
        self.w = window
        rec = window.rec
        self.setWindowTitle(f"Export {rec.name}")
        self.setMinimumWidth(560)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.out = QtWidgets.QLineEdit(str(default_output(rec.name)))
        browse = QtWidgets.QToolButton()
        browse.setIcon(theme.icon("open"))
        browse.setToolTip("Choose the output folder")
        browse.clicked.connect(self._browse)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.out, 1)
        row.addWidget(browse)
        form.addRow("Output folder", row)

        last = len(rec) - 1
        self.all_frames = QtWidgets.QRadioButton(f"All {len(rec)} frames")
        self.some_frames = QtWidgets.QRadioButton("Frames")
        self.first = QtWidgets.QSpinBox()
        self.last = QtWidgets.QSpinBox()
        for spin, value in ((self.first, window.index), (self.last, last)):
            spin.setRange(0, last)
            spin.setValue(value)
            spin.valueChanged.connect(lambda _=0: self.some_frames.setChecked(True))
        self.all_frames.setChecked(True)
        frames = QtWidgets.QHBoxLayout()
        frames.addWidget(self.all_frames)
        frames.addSpacing(12)
        frames.addWidget(self.some_frames)
        frames.addWidget(self.first)
        frames.addWidget(QtWidgets.QLabel("to"))
        frames.addWidget(self.last)
        frames.addStretch(1)
        form.addRow("Frames", frames)

        self.format = QtWidgets.QComboBox()
        for key, label in FORMAT_LABELS.items():
            self.format.addItem(label, key)
        self.compress = QtWidgets.QCheckBox("Compress (smaller files, slower)")
        self.compress.setChecked(True)
        self.format.currentIndexChanged.connect(
            lambda: self.compress.setEnabled(self.format.currentData() == "npz"))
        form.addRow("Format", self.format)
        form.addRow("", self.compress)

        self.workers = QtWidgets.QSpinBox()
        self.workers.setRange(1, os.cpu_count() or 4)
        self.workers.setValue(default_workers())
        self.workers.setToolTip("Frames processed at the same time")
        form.addRow("Threads", self.workers)

        steps = [s.name for s in window.pipeline if s.enabled]
        text = ("Steps: " + " → ".join(steps)) if steps else (
            "No steps: the frames are copied (or converted) as they are.")
        summary = QtWidgets.QLabel(text + "\nA frame whose steps fail is not written; "
                                   "export.log in the output folder lists it.")
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color: {theme.MUTED};")
        layout.addWidget(summary)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        export = buttons.addButton("Export", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        export.setDefault(True)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.overwrite = False

    def _browse(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder",
                                                          str(Path(self.out.text()).parent))
        if path:
            self.out.setText(path)

    def _accept(self) -> None:
        if not self.out.text().strip():
            return
        out = Path(self.out.text().strip())
        if out.resolve() == self.w.rec.path.resolve():
            QtWidgets.QMessageBox.warning(self, "Export", "Choose a folder other than the test "
                                          "run itself.")
            return
        if out.exists() and any(out.iterdir()):
            button = QtWidgets.QMessageBox.StandardButton
            answer = QtWidgets.QMessageBox.question(
                self, "Export", f"{out} is not empty. Replace the frames of an earlier export "
                "there? (other files stay)", button.Yes | button.No)
            if answer != button.Yes:
                return
            self.overwrite = True
        if self.some_frames.isChecked() and self.first.value() > self.last.value():
            QtWidgets.QMessageBox.warning(self, "Export", "The first frame is after the last.")
            return
        self.accept()

    def values(self) -> dict:
        frames = None
        if self.some_frames.isChecked():
            frames = range(self.first.value(), self.last.value() + 1)
        return {"out_dir": Path(self.out.text().strip()), "frames": frames,
                "fmt": self.format.currentData(), "workers": self.workers.value(),
                "compress": self.compress.isChecked(), "overwrite": self.overwrite}
