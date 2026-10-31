"""Information panel: file, time, pose, point counts, bounds and arrays of the current frame."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PySide6 import QtGui, QtWidgets

from . import theme


class InfoPanel(QtWidgets.QTreeWidget):
    """Sections of key/value rows, updated in place for every frame."""

    SECTIONS = ("File", "Time", "Pose", "Points", "Bounds", "Arrays", "Timing")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Property", "Value"])
        self.setUniformRowHeights(True)
        self.setRootIsDecorated(False)
        self.setIndentation(10)
        self.header().resizeSection(0, 120)
        self._sections: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._rows: dict[tuple[str, str], QtWidgets.QTreeWidgetItem] = {}
        bold = QtGui.QFont(self.font())
        bold.setBold(True)
        for name in self.SECTIONS:
            item = QtWidgets.QTreeWidgetItem(self, [name])
            item.setFont(0, bold)
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(theme.ACCENT)))
            item.setFirstColumnSpanned(True)
            self._sections[name] = item
        self.expandAll()

    def _set(self, section: str, key: str, value: str) -> None:
        row = self._rows.get((section, key))
        if row is None:
            row = QtWidgets.QTreeWidgetItem(self._sections[section], [key, value])
            self._rows[(section, key)] = row
        elif row.text(1) != value:
            row.setText(1, value)

    def _keep(self, section: str, keys: set[str]) -> None:
        for (sec, key), row in list(self._rows.items()):
            if sec == section and key not in keys:
                self._sections[section].removeChild(row)
                del self._rows[(sec, key)]

    def show_frame(self, window) -> None:
        w = window
        rec, raw, out, shown = w.rec, w.raw, w.output, w.shown
        if rec is None or raw is None:
            return
        path = Path(raw.meta.get("path", ""))
        self._set("File", "name", path.name)
        self._set("File", "folder", rec.name)
        if path.is_file():
            self._set("File", "size", f"{path.stat().st_size / 2**20:.1f} MB")
        layout = raw.meta.get("layout", "?")
        shape = raw.meta.get("shape")
        self._set("File", "layout", f"{layout} {shape[0]} × {shape[1]:,}" if shape else layout)

        self._set("Time", "frame", f"{w.index} / {len(rec) - 1}")
        t = raw.meta.get("timestamp")
        if t is not None:
            stamp = datetime.fromtimestamp(float(t), UTC)
            self._set("Time", "timestamp", stamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC")
            if w.t0 is not None:
                self._set("Time", "since start", f"{float(t) - w.t0:.3f} s")
        self._set("Time", "rate", f"{rec.rate_hz:.2f} Hz")

        pose = [(k, raw.meta[k]) for k in ("roll", "pitch", "yaw") if k in raw.meta]
        for key, value in pose:
            self._set("Pose", key, f"{np.degrees(float(value)):+.3f}°")
        self._sections["Pose"].setHidden(not pose)

        self._set("Points", "loaded", f"{len(raw):,}")
        self._set("Points", "after pipeline", f"{len(out):,}")
        self._set("Points", "shown", f"{len(shown):,}")
        echo_keys = set()
        if "echo" in raw.fields:
            values, counts = np.unique(raw.echo, return_counts=True)
            for e, c in zip(values.tolist(), counts.tolist(), strict=True):
                key = f"echo {e}"
                echo_keys.add(key)
                self._set("Points", key, f"{c:,}  ({100 * c / max(len(raw), 1):.1f} %)")
        self._keep("Points", {"loaded", "after pipeline", "shown", *echo_keys})

        lo, hi = shown.bounds()
        for k, axis in enumerate("xyz"):
            self._set("Bounds", axis, f"{lo[k]:.2f} … {hi[k]:.2f} m")

        keys = set()
        for name, values in out.fields.items():
            if values.ndim != 1 or not len(values) or values.dtype.kind not in "biuf":
                continue
            keys.add(name)
            vmin, vmax = np.nanmin(values), np.nanmax(values)
            self._set("Arrays", name, f"{values.dtype}  [{vmin:.4g}, {vmax:.4g}]")
        self._keep("Arrays", keys)

        load = rec.load_ms.get(w.index)
        if load is not None:
            self._set("Timing", "read file", f"{load:.1f} ms")
        for key, value in w.timing.items():
            self._set("Timing", key, f"{value:.1f} ms")
