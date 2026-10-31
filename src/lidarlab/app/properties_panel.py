"""Properties panel: the selected pipeline step's parameters, then how the current frame is
drawn (coloring, points, echoes, scene, camera)."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from . import theme
from .display import CATEGORICAL, COLORMAPS, SOLID
from .render_view import BACKGROUNDS, ECHO_COLORS
from .step_editor import StepEditor

CAMERA_BUTTONS = (("3D", "3d", "Chase view (1)"), ("BEV", "bev", "Bird's eye view (2)"),
                  ("Front", "front", "Driver's view (3)"), ("Side", "side", "Side view (4)"),
                  ("Fit", "fit", "Fit to the points (R)"))


class PropertiesPanel(QtWidgets.QScrollArea):
    """Edits ``window.display`` and asks the window to redraw."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.w = window
        self._syncing = False
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # rows shrink to the dock's width instead of hiding values behind a scroll bar
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(6, 0, 6, 6)

        self.step_editor = StepEditor(window)
        layout.addWidget(self.step_editor)

        coloring = QtWidgets.QGroupBox("Coloring")
        form = QtWidgets.QFormLayout(coloring)
        self.color_by = QtWidgets.QComboBox()
        self.color_by.setToolTip("Field the points are colored by (C cycles in the 3D view)")
        self.cmap = QtWidgets.QComboBox()
        self.cmap.setIconSize(QtCore.QSize(64, 12))
        for name in COLORMAPS:
            self.cmap.addItem(theme.colormap_icon(name), name)
        self.vmin, self.vmax = self._spin(), self._spin()
        rescale = QtWidgets.QToolButton()
        rescale.setIcon(theme.icon("refresh"))
        rescale.setToolTip("Rescale the color range to the current frame")
        rescale.clicked.connect(self.w.rescale)
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(self.vmin, 1)
        range_row.addWidget(QtWidgets.QLabel("–"))
        range_row.addWidget(self.vmax, 1)
        range_row.addWidget(rescale)
        self.legend = QtWidgets.QCheckBox("Show color legend")
        form.addRow("Color by", self.color_by)
        form.addRow("Colormap", self.cmap)
        form.addRow("Range", range_row)
        form.addRow("", self.legend)
        layout.addWidget(coloring)

        points = QtWidgets.QGroupBox("Points")
        form = QtWidgets.QFormLayout(points)
        self.point_size = QtWidgets.QDoubleSpinBox()
        self.point_size.setRange(1.0, 10.0)
        self.point_size.setSingleStep(0.5)
        self.point_size.setSuffix(" px")
        self.show_removed = QtWidgets.QCheckBox("Show removed points (red)")
        self.show_removed.setToolTip("Draw the points the pipeline removed, in red")
        self.split = QtWidgets.QCheckBox("Split view: raw | pipeline")
        self.split.setToolTip("Raw points on the left, the pipeline's output on the right, "
                              "one camera (Ctrl+T)")
        form.addRow("Size", self.point_size)
        form.addRow("", self.show_removed)
        form.addRow("", self.split)
        layout.addWidget(points)

        self.echo_box = QtWidgets.QGroupBox("Echoes")
        self.echo_layout = QtWidgets.QVBoxLayout(self.echo_box)
        self.echo_checks: dict[int, QtWidgets.QCheckBox] = {}
        layout.addWidget(self.echo_box)

        scene = QtWidgets.QGroupBox("Scene")
        form = QtWidgets.QFormLayout(scene)
        self.rings = QtWidgets.QCheckBox("Range rings (10 m)")
        self.grid = QtWidgets.QCheckBox("Ground grid (10 m)")
        self.ego = QtWidgets.QCheckBox("Ego vehicle outline")
        self.axes = QtWidgets.QCheckBox("Orientation axes")
        self.background = QtWidgets.QComboBox()
        self.background.addItems(list(BACKGROUNDS))
        for box in (self.rings, self.grid, self.ego, self.axes):
            form.addRow("", box)
        form.addRow("Background", self.background)
        layout.addWidget(scene)

        camera = QtWidgets.QGroupBox("Camera")
        row = QtWidgets.QHBoxLayout(camera)
        for label, preset, tip in CAMERA_BUTTONS:
            # tool buttons shrink to their text; five push buttons would need ~460 px
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Fixed)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, p=preset: self.w.set_camera(p))
            row.addWidget(button)
        layout.addWidget(camera)
        layout.addStretch(1)
        self.setWidget(content)

        self.color_by.currentTextChanged.connect(self._color_by_changed)
        self.cmap.currentTextChanged.connect(self._changed)
        self.vmin.valueChanged.connect(self._range_changed)
        self.vmax.valueChanged.connect(self._range_changed)
        for box in (self.legend, self.show_removed, self.split, self.rings, self.grid, self.ego,
                    self.axes):
            box.toggled.connect(self._changed)
        self.point_size.valueChanged.connect(self._changed)
        self.background.currentTextChanged.connect(self._changed)
        self.set_run([SOLID], [], {})

    @staticmethod
    def _spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(3)
        spin.setKeyboardTracking(False)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setMinimumWidth(60)  # the ±1e9 range would ask for ~130 px and widen the panel
        return spin

    def set_run(self, modes: list[str], echoes: list[int], labels: dict) -> None:
        """New test run: color-by choices and echo check boxes (labels from recording.json)."""
        self._syncing = True
        self.color_by.clear()
        self.color_by.addItems(modes)
        for box in self.echo_checks.values():
            box.setParent(None)  # off the panel now; deleteLater alone waits for the event loop
            box.deleteLater()
        self.echo_checks.clear()
        for e in echoes:
            label = labels.get(str(e), "")
            box = QtWidgets.QCheckBox(f"echo {e}" + (f" – {label}" if label else ""))
            box.setIcon(theme.dot_icon(ECHO_COLORS[e % len(ECHO_COLORS)]))
            box.toggled.connect(self._changed)
            self.echo_layout.addWidget(box)
            self.echo_checks[e] = box
        self.echo_box.setVisible(bool(echoes))
        self._syncing = False
        self.sync()

    def sync(self) -> None:
        """Show the window's current display settings."""
        d = self.w.display
        self._syncing = True
        self.color_by.setCurrentText(d.color_by)
        self.cmap.setCurrentText(d.cmap(d.color_by))
        lo, hi = d.clims.get(d.color_by, (0.0, 1.0))
        self.vmin.setValue(lo)
        self.vmax.setValue(hi)
        ranged = d.color_by not in (*CATEGORICAL, SOLID)
        for widget in (self.cmap, self.vmin, self.vmax):
            widget.setEnabled(ranged)
        self.legend.setChecked(d.legend)
        self.point_size.setValue(d.point_size)
        self.show_removed.setChecked(d.show_removed)
        self.split.setChecked(d.split)
        for e, box in self.echo_checks.items():
            box.setChecked(e not in d.hidden_echoes)
        self.rings.setChecked(d.rings)
        self.grid.setChecked(d.grid)
        self.ego.setChecked(d.ego)
        self.axes.setChecked(d.axes)
        self.background.setCurrentText(d.background)
        self._syncing = False

    def _color_by_changed(self, mode: str) -> None:
        if not self._syncing and mode:
            self.w.set_color_by(mode)

    def _range_changed(self) -> None:
        if self._syncing:
            return
        d = self.w.display
        if d.color_by not in (*CATEGORICAL, SOLID):
            d.clims[d.color_by] = (self.vmin.value(), self.vmax.value())
            self.w.apply_display()

    def _changed(self) -> None:
        if self._syncing:
            return
        d = self.w.display
        if d.color_by not in (*CATEGORICAL, SOLID):
            d.cmaps[d.color_by] = self.cmap.currentText()
        d.legend = self.legend.isChecked()
        d.point_size = self.point_size.value()
        d.show_removed = self.show_removed.isChecked()
        d.split = self.split.isChecked()
        d.hidden_echoes = {e for e, box in self.echo_checks.items() if not box.isChecked()}
        d.rings, d.grid = self.rings.isChecked(), self.grid.isChecked()
        d.ego, d.axes = self.ego.isChecked(), self.axes.isChecked()
        d.background = self.background.currentText()
        self.w.apply_display()
