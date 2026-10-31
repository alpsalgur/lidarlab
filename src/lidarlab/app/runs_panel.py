"""Test run browser: folders of .npz frames found under one or more root folders."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme

PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole


def find_runs(root: Path, depth: int = 3) -> list[Path]:
    """Folders under ``root`` (up to ``depth`` levels) that contain .npz files."""
    runs, level = [], [Path(root)]
    for _ in range(depth + 1):
        deeper = []
        for folder in level:
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            if any(e.name.endswith(".npz") and e.is_file() for e in entries):
                runs.append(folder)
            else:
                deeper += [Path(e.path) for e in entries
                           if e.is_dir() and not e.name.startswith(".")]
        level = deeper
    return sorted(runs)


def run_summary(folder: Path) -> dict:
    """Frame count, size on disk and what recording.json says about the run."""
    frames = size = 0
    for e in os.scandir(folder):
        if e.name.endswith(".npz") and e.is_file():
            frames += 1
            size += e.stat().st_size
    try:
        info = json.loads((folder / "recording.json").read_text())
    except (OSError, ValueError):
        info = {}
    dataset = info.get("dataset") or (
        "synthetic" if "synthetic" in info.get("generator", "") else info.get("source", ""))
    tip = [str(folder)]
    tip += [f"{key}: {info[key]}" for key in ("source", "sensor", "layout", "license")
            if key in info]
    tip += [f"echo {k}: {v}" for k, v in info.get("echoes", {}).items()]
    return {"frames": frames, "size": size, "dataset": dataset, "tooltip": "\n".join(tip)}


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class RunsPanel(QtWidgets.QWidget):
    """Lists the test runs under the root folders; double-click opens one."""

    openRequested = QtCore.Signal(str)

    def __init__(self, settings: QtCore.QSettings, default_roots: list[Path], parent=None):
        super().__init__(parent)
        self.settings = settings
        self.default_roots = [str(p) for p in default_roots if p.is_dir()]
        self.current: str | None = None
        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.addAction(theme.icon("add"), "Add a folder to search for test runs", self.add_root)
        toolbar.addAction(theme.icon("refresh"), "Rescan the folders", self.scan)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Test run", "Frames", "Size", "Dataset"])
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.tree)
        self.scan()

    @property
    def roots(self) -> list[str]:
        roots = self.settings.value("runs/roots")
        if roots is None:
            return list(self.default_roots)
        return [roots] if isinstance(roots, str) else list(roots)

    def add_root(self, path: str | None = None) -> None:
        if not path:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with test runs")
        if path and path not in self.roots:
            self.settings.setValue("runs/roots", [*self.roots, path])
            self.scan()

    def remove_root(self, path: str) -> None:
        self.settings.setValue("runs/roots", [r for r in self.roots if r != path])
        self.scan()

    def scan(self) -> None:
        self.tree.clear()
        for root in self.roots:
            root_path = Path(root)
            root_item = QtWidgets.QTreeWidgetItem([root_path.name or root])
            root_item.setIcon(0, theme.icon("folder", theme.MUTED))
            root_item.setToolTip(0, root)
            root_item.setData(0, PATH_ROLE, ("root", root))
            self.tree.addTopLevelItem(root_item)
            groups: dict[Path, QtWidgets.QTreeWidgetItem] = {}
            for run in find_runs(root_path):
                parent = root_item
                rel = run.parent.relative_to(root_path) if run != root_path else Path(".")
                for part in [p for p in rel.parts if p]:
                    key = (parent, part)
                    if key not in groups:
                        groups[key] = QtWidgets.QTreeWidgetItem(parent, [part])
                        groups[key].setIcon(0, theme.icon("folder", theme.MUTED))
                    parent = groups[key]
                s = run_summary(run)
                item = QtWidgets.QTreeWidgetItem(
                    parent, [run.name, str(s["frames"]), human_size(s["size"]), s["dataset"]])
                item.setIcon(0, theme.icon("tape", theme.ACCENT))
                item.setData(0, PATH_ROLE, ("run", str(run)))
                for col in range(4):
                    item.setToolTip(col, s["tooltip"])
                for col in (1, 2):
                    item.setTextAlignment(col, QtCore.Qt.AlignmentFlag.AlignRight
                                          | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.tree.expandAll()
        for col in range(4):
            self.tree.resizeColumnToContents(col)
        self.set_current(self.current)

    def set_current(self, path: str | None) -> None:
        """Show the open test run in bold."""
        self.current = path
        target = str(Path(path)) if path else None
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            data = item.data(0, PATH_ROLE)
            font = item.font(0)
            font.setBold(bool(data and data[0] == "run" and str(Path(data[1])) == target))
            item.setFont(0, font)
            iterator += 1

    def _double_clicked(self, item: QtWidgets.QTreeWidgetItem) -> None:
        data = item.data(0, PATH_ROLE)
        if data and data[0] == "run":
            self.openRequested.emit(data[1])

    def _context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.tree.itemAt(pos)
        data = item.data(0, PATH_ROLE) if item else None
        if not data:
            return
        kind, path = data
        menu = QtWidgets.QMenu(self)
        if kind == "run":
            menu.addAction(theme.icon("open"), "Open", lambda: self.openRequested.emit(path))
        menu.addAction("Show in Explorer",
                       lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path)))
        if kind == "root":
            menu.addAction(theme.icon("remove"), "Remove from list", lambda: self.remove_root(path))
        menu.exec(self.tree.viewport().mapToGlobal(pos))
