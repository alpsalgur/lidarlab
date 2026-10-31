"""Parameters of the selected pipeline step, as sliders, boxes and lists that apply live."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from lidarlab import Param

from . import theme

FIELD_EXTRAS = ("x", "y", "z", "range", "azimuth", "elevation")
SLIDER_STEPS = 1000


def _reflow(text: str) -> str:
    """Docstring lines joined into paragraphs, so the label wraps them to its own width."""
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    return "\n\n".join(p for p in paragraphs if p)


def _decimals(p: Param) -> int:
    """Decimals that show the parameter's step (0.01 -> 2, 0.005 -> 3), else 3."""
    return max(0, min(6, -math.floor(math.log10(p.step)))) if p.step else 3


class ParamWidget(QtWidgets.QWidget):
    """Edits one parameter; emits ``changed(value)`` with the checked value."""

    changed = QtCore.Signal(object)

    def __init__(self, param: Param, field_names: list[str], parent=None):
        super().__init__(parent)
        self.param = p = param
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = None
        if p.kind == "bool":
            self.editor = QtWidgets.QCheckBox()
            self.editor.toggled.connect(lambda on: self.changed.emit(on))
        elif p.kind in ("choice", "field"):
            self.editor = QtWidgets.QComboBox()
            items = list(p.choices) if p.kind == "choice" else field_names
            self.editor.addItems(items)
            if p.kind == "field":
                self.editor.setEditable(True)
                self.editor.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            self.editor.currentTextChanged.connect(lambda text: text and self.changed.emit(text))
        else:
            integer = p.kind == "int"
            self.editor = QtWidgets.QSpinBox() if integer else QtWidgets.QDoubleSpinBox()
            lo = -1e9 if p.lo is None else p.lo
            hi = 1e9 if p.hi is None else p.hi
            if integer:
                self.editor.setRange(int(max(lo, -2**31)), int(min(hi, 2**31 - 1)))
                self.editor.setSingleStep(int(p.step or 1))
            else:
                self.editor.setDecimals(_decimals(p))
                self.editor.setRange(lo, hi)
                bounded = p.lo is not None and p.hi is not None
                self.editor.setSingleStep(p.step or ((hi - lo) / 100 if bounded else 0.1))
            self.editor.setKeyboardTracking(False)
            if p.unit:
                self.editor.setSuffix(f" {p.unit}")
            self.editor.valueChanged.connect(self._spin_changed)
            if p.lo is not None and p.hi is not None:
                self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                self.slider.setRange(0, int(p.hi - p.lo) if integer else SLIDER_STEPS)
                self.slider.valueChanged.connect(self._slider_changed)
                self.slider.setMinimumWidth(40)
                layout.addWidget(self.slider, 3)
                self.editor.setFixedWidth(84)
        layout.addWidget(self.editor, 2 if self.slider else 1)
        if p.help:
            self.setToolTip(p.help)

    def set_value(self, value) -> None:
        """Show ``value`` without emitting ``changed``."""
        blockers = [QtCore.QSignalBlocker(w) for w in (self.editor, self.slider) if w is not None]
        p = self.param
        if p.kind == "bool":
            self.editor.setChecked(bool(value))
        elif p.kind in ("choice", "field"):
            if self.editor.findText(str(value)) < 0:
                self.editor.addItem(str(value))
            self.editor.setCurrentText(str(value))
        else:
            self.editor.setValue(value)
            if self.slider is not None:
                self.slider.setValue(self._to_slider(value))
        del blockers

    def _to_slider(self, value) -> int:
        p = self.param
        if p.kind == "int":
            return int(value - p.lo)
        return round((value - p.lo) / (p.hi - p.lo) * SLIDER_STEPS)

    def _slider_changed(self, position: int) -> None:
        p = self.param
        if p.kind == "int":
            value = p.lo + position
        else:
            value = p.lo + (p.hi - p.lo) * position / SLIDER_STEPS
        if p.step and p.kind == "float":
            value = round(round((value - p.lo) / p.step) * p.step + p.lo, _decimals(p))
        with QtCore.QSignalBlocker(self.editor):
            self.editor.setValue(value)
        self.changed.emit(p.check(value))

    def _spin_changed(self, value) -> None:
        if self.slider is not None:
            with QtCore.QSignalBlocker(self.slider):
                self.slider.setValue(self._to_slider(value))
        self.changed.emit(self.param.check(value))


class StepEditor(QtWidgets.QGroupBox):
    """Parameters of one pipeline step. Changes apply to the current frame right away; slider
    drags are throttled so a slow filter does not queue up work."""

    def __init__(self, window, parent=None):
        super().__init__("Step", parent)
        self.w = window
        self.step_name: str | None = None
        self.widgets: dict[str, ParamWidget] = {}
        self._pending: dict = {}
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._apply)

        layout = QtWidgets.QVBoxLayout(self)
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet(f"color: {theme.MUTED};")
        self.description.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.description)
        self.enabled = QtWidgets.QCheckBox("Enabled")
        self.enabled.toggled.connect(self._enabled_toggled)
        layout.addWidget(self.enabled)
        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.form_host)
        self.no_params = QtWidgets.QLabel("A plain function: give it parameters with @lab.filter "
                                          "to get sliders here.")
        self.no_params.setWordWrap(True)
        self.no_params.setStyleSheet(f"color: {theme.MUTED};")
        layout.addWidget(self.no_params)
        self.stats = QtWidgets.QLabel()
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)
        buttons = QtWidgets.QHBoxLayout()
        self.reset_button = QtWidgets.QPushButton("Reset to defaults")
        self.reset_button.clicked.connect(self._reset)
        self.source_button = QtWidgets.QPushButton("Open source")
        self.source_button.clicked.connect(self._open_source)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.source_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.setVisible(False)

    def show_step(self, name: str | None) -> None:
        """Edit step ``name`` (None hides the editor)."""
        self._flush()
        if name is not None and name not in self.w.pipeline:
            name = None
        self.step_name = name
        while self.form.rowCount():
            self.form.removeRow(0)
        self.widgets.clear()
        if name is None:
            self.setVisible(False)
            return
        step = self.w.pipeline[name]
        title = getattr(step.func, "title", "") or ""
        self.setTitle(f"Step: {name}")
        text = _reflow(step.description)
        if title and title != name:
            text = f"{title}\n\n{text}" if text else title
        self.description.setText(text)
        self.description.setVisible(bool(text))
        fields = self._field_names()
        for key, param in step.params.items():
            widget = ParamWidget(param, fields)
            widget.changed.connect(lambda value, k=key: self._edited(k, value))
            label = QtWidgets.QLabel(key)
            if param.help:
                label.setToolTip(param.help)
            self.form.addRow(label, widget)
            self.widgets[key] = widget
        self.no_params.setVisible(not step.params)
        self.reset_button.setVisible(bool(step.params))
        self.source_button.setVisible(step.source is not None)
        self.setVisible(True)
        self.sync()

    def _field_names(self) -> list[str]:
        raw = self.w.raw
        names = [k for k, v in raw.fields.items() if v.ndim == 1] if raw is not None else []
        return names + [n for n in FIELD_EXTRAS if n not in names]

    def sync(self) -> None:
        """Show the step's current values and stats (after changes from scripts)."""
        if self.step_name is None:
            return
        if self.step_name not in self.w.pipeline:
            self.show_step(None)
            return
        step = self.w.pipeline[self.step_name]
        values = step.values
        for key, widget in self.widgets.items():
            if key in values and key not in self._pending:
                widget.set_value(values[key])
        with QtCore.QSignalBlocker(self.enabled):
            self.enabled.setChecked(step.enabled)
        self.update_stats()

    def update_stats(self) -> None:
        if self.step_name is None or self.step_name not in self.w.pipeline:
            return
        s = self.w.pipeline[self.step_name]
        if s.error is not None:
            text = f"<span style='color:{theme.ERROR}'>{type(s.error).__name__}: {s.error}</span>"
        elif not s.enabled:
            text = "off"
        elif s.n_out is None:
            text = ""
        else:
            removed = s.n_in - s.n_out
            share = 100 * removed / max(s.n_in, 1)
            text = (f"{s.n_in:,} → {s.n_out:,} points"
                    + (f" (−{removed:,}, {share:.1f} %)" if removed else "")
                    + f" · {s.ms:.0f} ms")
        self.stats.setText(text)

    # editing

    def _edited(self, key: str, value) -> None:
        self._pending[key] = value
        if not self._timer.isActive():
            self._timer.start()

    def _apply(self) -> None:
        if not self._pending or self.step_name not in self.w.pipeline:
            self._pending.clear()
            return
        values, self._pending = self._pending, {}
        name = self.step_name
        self.w.apply_live(lambda: self.w.pipeline.set(name, **values))

    def _flush(self) -> None:
        self._timer.stop()
        self._apply()

    def _enabled_toggled(self, on: bool) -> None:
        if self.step_name in self.w.pipeline:
            self.w.pipeline.enable(self.step_name, on)

    def _reset(self) -> None:
        if self.step_name in self.w.pipeline:
            self._pending.clear()
            self.w.pipeline[self.step_name].reset()

    def _open_source(self) -> None:
        if self.step_name in self.w.pipeline:
            source = self.w.pipeline[self.step_name].source
            if source and not source[0].startswith("<"):
                self.w.editor.open_file(source[0], line=source[1])
                self.w.docks["editor"].raise_()


def filter_menu(parent: QtWidgets.QWidget, add) -> QtWidgets.QMenu:
    """Menu of the built-in filters by group; ``add(key, name, values)`` adds one."""
    from lidarlab.core.filters import available

    menu = QtWidgets.QMenu("Add Filter", parent)
    menu.setIcon(theme.icon("add"))
    groups: dict[str, QtWidgets.QMenu] = {}
    for key, cls in available().items():
        group = groups.get(cls.group)
        if group is None:
            group = groups[cls.group] = menu.addMenu(cls.group)
        action = group.addAction(cls.title or key, lambda k=key: add(k, None, {}))
        action.setToolTip(cls.__doc__.split("\n")[0] if cls.__doc__ else "")
    menu.addSeparator()
    presets = (
        ("Range limits", "threshold", "range_limits", {"field": "range", "min": 0.5, "max": 120.0}),
        ("Height band", "threshold", "height_band", {"field": "z", "min": -3.0, "max": 4.0}),
        ("Remove own vehicle", "crop_box", "own_vehicle",
         {"x_min": -3.0, "x_max": 2.5, "y_min": -1.2, "y_max": 1.2, "z_min": -2.5, "z_max": 0.5,
          "mode": "remove inside"}),
    )
    for label, key, name, values in presets:
        menu.addAction(label, lambda k=key, n=name, v=values: add(k, n, v))
    for group in groups.values():
        group.setToolTipsVisible(True)
    return menu


def preset_dialog(parent, save: bool, start: str) -> str:
    """File dialog for pipeline presets; returns the path or an empty string."""
    kind = "YAML preset (*.yaml *.yml)"
    if save:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Save pipeline preset", start, kind)
    else:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, "Load pipeline preset", start, kind)
    return path
