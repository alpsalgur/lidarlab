"""Pipeline browser: the open test run, its processing steps, highlight and overlay layers."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from lidarlab import Filter
from lidarlab.core.filter import FunctionFilter

from . import theme
from .step_editor import filter_menu

KIND_ROLE = QtCore.Qt.ItemDataRole.UserRole
RIGHT = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
CHECKED = QtCore.Qt.CheckState.Checked
UNCHECKED = QtCore.Qt.CheckState.Unchecked


class PipelinePanel(QtWidgets.QWidget):
    """Tree of what is drawn: the run (raw points), its steps in order, then the layers.

    Check boxes switch steps and layers on and off; the columns show the points left after
    each step (with the change) and how long the step took on the current frame. Selecting a
    step shows its parameters under Properties.
    """

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.w = window
        self._building = False
        self._signature = None
        self._step_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._layer_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self.run_item: QtWidgets.QTreeWidgetItem | None = None
        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(16, 16))
        self.add_menu = filter_menu(self, window.add_filter)
        add = QtWidgets.QToolButton()
        add.setMenu(self.add_menu)
        add.setIcon(theme.icon("add"))
        add.setText("Filter")
        add.setToolTip("Add a built-in filter after the last step")
        add.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        add.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(add)
        toolbar.addAction(theme.icon("open"), "Load pipeline preset…", window.load_preset)
        toolbar.addAction(theme.icon("save"), "Save pipeline preset…", window.save_preset)
        toolbar.addSeparator()
        toolbar.addAction(theme.icon("up"), "Move step up", lambda: self._move(-1))
        toolbar.addAction(theme.icon("down"), "Move step down", lambda: self._move(1))
        toolbar.addAction(theme.icon("remove"), "Remove the selected step or layer", self._remove)
        toolbar.addSeparator()
        toolbar.addAction(theme.icon("trash"), "Remove all steps and layers", self._clear)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Points", "ms"])
        self.tree.setUniformRowHeights(True)
        self.tree.setRootIsDecorated(True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.resizeSection(1, 112)
        header.resizeSection(2, 44)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemDoubleClicked.connect(self._open_source)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.tree)

    def rebuild(self) -> None:
        """Rebuild the tree if steps or layers were added, removed, reordered or toggled;
        otherwise just refresh the numbers. The selection survives."""
        w = self.w
        signature = (w.rec.name if w.rec is not None else None, w.view.cloud.visible,
                     tuple((s.name, s.enabled, id(s.func)) for s in w.pipeline),
                     tuple((n, layer.visible, layer.color) for n, layer in w.layers.items()))
        if signature == self._signature:
            self.update_stats()
            return
        self._signature = signature
        selected = self._selected()
        self._building = True
        self.tree.clear()
        self._step_items.clear()
        self._layer_items.clear()
        name = w.rec.name if w.rec is not None else "(no test run)"
        self.run_item = self._item(None, name, ("run", None), w.view.cloud.visible,
                                   theme.icon("tape", theme.ACCENT))
        tip = str(w.rec.path) if w.rec is not None else "File > Open Test Run"
        self.run_item.setToolTip(0, tip)
        for step in w.pipeline:
            builtin = isinstance(step.func, Filter) and not isinstance(step.func, FunctionFilter)
            icon = theme.text_icon("F", theme.ACCENT) if builtin else theme.text_icon("ƒ")
            item = self._item(self.run_item, step.name, ("step", step.name), step.enabled, icon)
            title = getattr(step.func, "title", "")
            doc = step.description.split("\n\n")[0]
            source = step.source
            tip = [f"{step.name}" + (f" – {title}" if title else "")] + ([doc] if doc else [])
            if source and not builtin:
                tip.append(f"{source[0]}:{source[1]}  (double-click to open)")
            item.setToolTip(0, "\n".join(tip))
            self._step_items[step.name] = item
        for layer in w.layers.values():
            kind = "highlight" if layer.func is not None else "overlay"
            item = self._item(None, layer.name, ("layer", layer.name), layer.visible,
                              theme.dot_icon(layer.color))
            item.setToolTip(0, f"{kind} layer, {layer.color}")
            self._layer_items[layer.name] = item
        self.tree.expandAll()
        if selected is not None:
            kind, key = selected
            item = (self._step_items.get(key) if kind == "step" else
                    self._layer_items.get(key) if kind == "layer" else self.run_item)
            if item is not None:
                self.tree.setCurrentItem(item)
        self._building = False
        if selected is not None and selected[0] == "step":
            # the step may be gone, or redefined by a script with other parameters
            w.properties.step_editor.show_step(selected[1] if selected[1] in w.pipeline else None)
        self.update_stats()

    def select_step(self, name: str) -> None:
        item = self._step_items.get(name)
        if item is not None:
            self.tree.setCurrentItem(item)

    def _current_changed(self, item, _previous=None) -> None:
        if self._building:
            return
        kind, name = item.data(0, KIND_ROLE) if item is not None else (None, None)
        self.w.properties.step_editor.show_step(name if kind == "step" else None)

    def _item(self, parent, text, kind, checked, icon) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([text, "", ""])
        item.setData(0, KIND_ROLE, kind)
        item.setIcon(0, icon)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, CHECKED if checked else UNCHECKED)
        item.setTextAlignment(1, RIGHT)
        item.setTextAlignment(2, RIGHT)
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        return item

    def update_stats(self) -> None:
        """Refresh the point counts, timings and error marks for the current frame."""
        w = self.w
        if self.run_item is None:
            return
        self._building = True
        muted = QtGui.QBrush(QtGui.QColor(theme.MUTED))
        normal = QtGui.QBrush(QtGui.QColor(theme.TEXT))
        if w.raw is not None:
            self.run_item.setText(1, f"{len(w.raw):,}")
            load = w.rec.load_ms.get(w.index) if w.rec is not None else None
            self.run_item.setText(2, f"{load:.0f}" if load is not None else "")
            self.run_item.setToolTip(2, "time to read the frame file, ms")
        for step in w.pipeline:
            item = self._step_items.get(step.name)
            if item is None:
                continue
            if step.error is not None:
                item.setText(1, "error")
                item.setText(2, "")
                item.setForeground(0, QtGui.QBrush(QtGui.QColor(theme.ERROR)))
                item.setForeground(1, QtGui.QBrush(QtGui.QColor(theme.ERROR)))
                item.setToolTip(1, f"{type(step.error).__name__}: {step.error}")
                continue
            item.setForeground(0, normal if step.enabled else muted)
            item.setForeground(1, normal if step.enabled else muted)
            item.setToolTip(1, "")
            if step.enabled and step.n_out is not None:
                delta = step.n_out - step.n_in
                change = f" ({delta / 1000:+.1f}k)" if delta else ""
                item.setText(1, f"{step.n_out:,}{change}")
                item.setToolTip(1, f"{step.n_in:,} in, {step.n_out:,} out ({delta:+,})")
                item.setText(2, f"{step.ms:.1f}")
            else:
                item.setText(1, "off" if not step.enabled else "")
                item.setText(2, "")
        for name, item in self._layer_items.items():
            layer = w.layers.get(name)
            if layer is None:
                continue
            failed = layer.error is not None
            item.setText(1, "error" if failed else f"{layer.count:,}")
            item.setForeground(1, QtGui.QBrush(QtGui.QColor(theme.ERROR if failed else theme.TEXT)))
            item.setToolTip(1, f"{type(layer.error).__name__}: {layer.error}" if failed else "")
        self._building = False
        w.properties.step_editor.update_stats()

    def _selected(self) -> tuple[str, str | None] | None:
        item = self.tree.currentItem()
        return item.data(0, KIND_ROLE) if item else None

    def _item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if self._building or column != 0:
            return
        kind, name = item.data(0, KIND_ROLE)
        on = item.checkState(0) == CHECKED
        if kind == "run":
            self.w.set_cloud_visible(on)
        elif kind == "step":
            self.w.pipeline.enable(name, on)
        elif kind == "layer":
            self.w.set_layer_visible(name, on)

    def _move(self, offset: int) -> None:
        selected = self._selected()
        if selected and selected[0] == "step":
            self.w.pipeline.move(selected[1], offset)
            self.tree.setCurrentItem(self._step_items.get(selected[1]))

    def _remove(self) -> None:
        selected = self._selected()
        if not selected:
            return
        kind, name = selected
        if kind == "step":
            self.w.pipeline.remove(name)
        elif kind == "layer":
            self.w.remove_layer(name)

    def _clear(self) -> None:
        self.w.clear_layers()
        self.w.pipeline.clear()

    def _open_source(self, item: QtWidgets.QTreeWidgetItem) -> None:
        kind, name = item.data(0, KIND_ROLE)
        if kind == "step" and name in self.w.pipeline:
            source = self.w.pipeline[name].source
            if source and not source[0].startswith("<"):
                self.w.editor.open_file(source[0], line=source[1])
                self.w.docks["editor"].raise_()

    def _context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        kind, _ = item.data(0, KIND_ROLE)
        menu = QtWidgets.QMenu(self)
        if kind == "step":
            menu.addAction(theme.icon("up"), "Move up", lambda: self._move(-1))
            menu.addAction(theme.icon("down"), "Move down", lambda: self._move(1))
            menu.addAction("Open source", lambda: self._open_source(item))
        if kind in ("step", "layer"):
            menu.addAction(theme.icon("remove"), "Remove", self._remove)
        menu.addSeparator()
        menu.addAction(theme.icon("trash"), "Remove all steps and layers", self._clear)
        menu.exec(self.tree.viewport().mapToGlobal(pos))
