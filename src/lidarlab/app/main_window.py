"""Main window: docks around the 3D view and a video-style player bar."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import lidarlab
from lidarlab import Frame, Processor, Recording, write_frame
from lidarlab.core.filters import available

from . import theme
from .api import HELP_HTML, ViewAPI
from .console import PythonShell
from .display import (
    CATEGORICAL,
    SOLID,
    Display,
    auto_range,
    color_modes,
    color_values,
    estimate_ground,
    removed_points,
)
from .editor import ScriptEditor
from .info_panel import InfoPanel
from .paths import data_dir, examples_dir, scripts_dir
from .pipeline_panel import PipelinePanel
from .player import Layer, Player, Rate
from .properties_panel import PropertiesPanel
from .render_view import LABEL_COLORS, RenderView, category_color, category_lut, colormap_lut
from .runs_panel import RunsPanel
from .step_editor import filter_menu, preset_dialog
from .timeline import SPEEDS, PlayerBar

Qt = QtCore.Qt
REMOVED = "__removed__"  # view layer showing the points the steps removed
MAX_RECENT = 8
# keys typed into these widgets stay with them instead of driving playback
TEXT_INPUTS = (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit,
               QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QAbstractItemView)

KEYS_HTML = """
<h3>Keyboard</h3>
<p>Playback keys work everywhere except while typing in the shell, the editor or a field.</p>
<table cellpadding="3">
<tr><td><b>Space</b></td><td>play / pause</td></tr>
<tr><td><b>← / →</b></td><td>previous / next frame (with Shift: 10 frames)</td></tr>
<tr><td><b>Home / End</b></td><td>first / last frame</td></tr>
<tr><td><b>1 2 3 4</b></td><td>camera: 3D, bird's eye, driver, side</td></tr>
<tr><td><b>R</b></td><td>fit the camera to the points</td></tr>
<tr><td><b>C</b></td><td>next color-by field</td></tr>
<tr><td><b>+ / −</b></td><td>point size</td></tr>
<tr><td><b>F5</b></td><td>run the script in the editor</td></tr>
<tr><td><b>Ctrl+Enter</b></td><td>run the selection or line in the editor</td></tr>
<tr><td><b>F1</b></td><td>scripting reference</td></tr>
</table>
<h3>Mouse in the 3D view</h3>
<p>Left drag: rotate · middle drag or Shift+left drag: pan · wheel or right drag: zoom.</p>
"""


class MainWindow(QtWidgets.QMainWindow):
    """Test runs and pipeline on the left, properties and information on the right,
    Python shell and script editor below, the 3D view and player bar in the middle."""

    frameLoaded = QtCore.Signal(int)  # emitted by loader threads, handled in the GUI thread
    frameProcessed = QtCore.Signal(int)  # emitted by processing threads

    def __init__(self, run: str | None = None, restore_layout: bool = True):
        super().__init__()
        self.setWindowTitle("LidarLab")
        self.setWindowIcon(theme.app_icon())
        self.resize(1600, 1000)
        self.settings = QtCore.QSettings()
        self.rec: Recording | None = None
        self.index = 0
        self.raw: Frame | None = None
        self.output: Frame | None = None
        self.shown: Frame | None = None
        self.t0: float | None = None  # timestamp of frame 0
        self.pipeline = lidarlab.pipeline
        self.processor = Processor(self.pipeline)  # steps run ahead of the playhead
        self.processor.on_done = self.frameProcessed.emit
        self.preview_pending: int | None = None  # frame waiting for its background result
        self._live = False  # set while a slider applies a change
        self.display = Display()
        self.layers: dict[str, Layer] = {}
        self.timing: dict[str, float] = {}
        self.fps = Rate()
        self._wanted: int | None = None
        self._reported: dict[str, str | None] = {}
        self._last_info = 0.0
        self._cache_version = -1
        self._processed_version = -1

        self.view = RenderView(self)
        self.player = Player(self)
        self.runs_panel = RunsPanel(self.settings, [data_dir()])
        self.pipeline_panel = PipelinePanel(self)
        self.properties = PropertiesPanel(self)
        self.info = InfoPanel()
        self.api = ViewAPI(self)
        namespace = {"lab": lidarlab, "np": np, "view": self.api, "pipe": self.pipeline,
                     "rec": None, "frame": None}
        self.shell = PythonShell(namespace)
        self.editor = ScriptEditor(lambda: self.shell.namespace, scripts_dir())
        self._make_actions()
        self.player_bar = PlayerBar(self.act, self)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.player_bar)
        self.setCentralWidget(central)
        self._make_docks()
        self._make_menus()
        self._make_toolbar()
        self._make_status_bar()

        self.runs_panel.openRequested.connect(self.open_run)
        self.editor.runFile.connect(self.shell.run_file)
        self.editor.runCode.connect(self.shell.execute)
        self.player_bar.seek.connect(self.show_frame)
        self.player_bar.speedChanged.connect(self.player.set_speed)
        self.player.stateChanged.connect(self._play_state_changed)
        self.frameLoaded.connect(self._frame_loaded)
        self.frameProcessed.connect(self._frame_processed)
        self.pipeline.subscribe(self._pipeline_changed)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self._excepthook = sys.excepthook
        sys.excepthook = self._report_exception
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(250)
        self._poll.timeout.connect(self._poll_status)
        self._poll.start()

        self._restore(restore_layout)
        self.apply_display()
        self.pipeline_panel.rebuild()
        start = run or self.settings.value("lastRun")
        if start and Path(start).is_dir():
            QtCore.QTimer.singleShot(0, lambda: self.open_run(start))

    # construction

    def _make_actions(self) -> None:
        def action(text, slot, shortcut=None, icon=None, tip=None, checkable=False, icon_text=None):
            a = QtGui.QAction(text, self)
            if icon:
                a.setIcon(theme.icon(icon))
            if shortcut:
                a.setShortcut(QtGui.QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
                a.setStatusTip(tip)
            if icon_text:
                a.setIconText(icon_text)
            a.setCheckable(checkable)
            a.triggered.connect(slot)
            self.addAction(a)
            return a

        def last() -> None:
            if self.rec is not None:
                self.show_frame(len(self.rec) - 1)

        self.act = {
            "open": action("&Open Test Run…", self.open_run_dialog, "Ctrl+O", "open",
                           "Open a folder of .npz frames"),
            "add_root": action("Add Test Run &Folder…", lambda: self.runs_panel.add_root(),
                               icon="add"),
            "export": action("&Export Current Frame…", self.export_frame, "Ctrl+E", "save",
                             "Save the current frame after the processing steps as .npz"),
            "screenshot": action("Save View &Screenshot…", self.save_screenshot, "F12", "camera"),
            "quit": action("E&xit", self.close, "Ctrl+Q"),
            "play": action("Play / Pause\tSpace", self.player.toggle, icon="play",
                           tip="Play / pause (Space)"),
            "first": action("First Frame\tHome", lambda: self.show_frame(0), icon="first",
                            tip="First frame (Home)"),
            "prev": action("Previous Frame\tLeft", lambda: self.step(-1), icon="prev",
                           tip="Previous frame (←)"),
            "next": action("Next Frame\tRight", lambda: self.step(1), icon="next",
                           tip="Next frame (→)"),
            "last": action("Last Frame\tEnd", last, icon="last", tip="Last frame (End)"),
            "loop": action("Loop", lambda on: setattr(self.player, "loop", on), icon="loop",
                           tip="Loop playback", checkable=True),
            "cache_all": action("Load All Frames into Memory", self.cache_all, icon="memory",
                                tip="Read all frames in the background, as far as memory allows"),
            "cam_fit": action("Fit to Points\tR", lambda: self.set_camera("fit"), icon="fit",
                              tip="Fit the camera to the points (R)"),
            "rescale": action("Rescale Colors to Frame", self.rescale, icon="refresh",
                              tip="Fit the color range to the current frame"),
            "clear_pipeline": action("Remove All Steps", lambda: self.pipeline.clear(),
                                     icon="trash"),
            "split": action("Split View: Raw | After Pipeline", self.toggle_split, "Ctrl+T",
                            "split", "Raw points left, the pipeline's output right (Ctrl+T)",
                            checkable=True),
            "load_preset": action("&Load Pipeline Preset…", self.load_preset, icon="open"),
            "save_preset": action("&Save Pipeline Preset…", self.save_preset, icon="save"),
            "clear_layers": action("Remove Highlights and Overlays", self.clear_layers),
            "reset_layout": action("Reset Layout", self.reset_layout),
            "help_api": action("&Scripting Reference", self.show_scripting_help, "F1"),
            "help_keys": action("&Keyboard and Mouse", self.show_keys_help),
            "about": action("&About LidarLab", self.show_about),
        }
        cameras = (("3d", "3D View\t1", "3D", "Chase view (1)"),
                   ("bev", "Bird's Eye View\t2", "BEV", "Bird's eye view (2)"),
                   ("front", "Driver View\t3", "Front", "Driver's view (3)"),
                   ("side", "Side View\t4", "Side", "Side view (4)"))
        for preset, text, short, tip in cameras:
            self.act[f"cam_{preset}"] = action(text, lambda _=False, p=preset: self.set_camera(p),
                                               tip=tip, icon_text=short)
        self.act["loop"].setChecked(True)

    def _make_docks(self) -> None:
        def dock(title, widget, name):
            d = QtWidgets.QDockWidget(title, self)
            d.setObjectName(name)
            d.setWidget(widget)
            return d

        self.docks = {
            "runs": dock("Test Runs", self.runs_panel, "TestRunsDock"),
            "pipeline": dock("Pipeline Browser", self.pipeline_panel, "PipelineBrowserDock"),
            "properties": dock("Properties", self.properties, "PropertiesDock"),
            "info": dock("Information", self.info, "InformationDock"),
            "shell": dock("Python Shell", self.shell, "PythonShellDock"),
            "editor": dock("Script Editor", self.editor, "ScriptEditorDock"),
        }
        d, area = self.docks, Qt.DockWidgetArea
        self.setCorner(Qt.Corner.BottomLeftCorner, area.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, area.RightDockWidgetArea)
        self.addDockWidget(area.LeftDockWidgetArea, d["runs"])
        self.splitDockWidget(d["runs"], d["pipeline"], Qt.Orientation.Vertical)
        self.addDockWidget(area.RightDockWidgetArea, d["properties"])
        self.splitDockWidget(d["properties"], d["info"], Qt.Orientation.Vertical)
        self.addDockWidget(area.BottomDockWidgetArea, d["shell"])
        self.splitDockWidget(d["shell"], d["editor"], Qt.Orientation.Horizontal)
        self._default_state = self.saveState()

    def _default_sizes(self) -> None:
        d = self.docks
        horizontal, vertical = Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        self.resizeDocks([d["runs"], d["properties"]], [340, 390], horizontal)
        self.resizeDocks([d["runs"], d["pipeline"]], [400, 400], vertical)
        self.resizeDocks([d["properties"], d["info"]], [480, 320], vertical)
        self.resizeDocks([d["shell"]], [330], vertical)
        self.resizeDocks([d["shell"], d["editor"]], [500, 500], horizontal)

    def _make_menus(self) -> None:
        a, bar = self.act, self.menuBar()
        file = bar.addMenu("&File")
        file.addAction(a["open"])
        self.recent_menu = file.addMenu("Recent Test Runs")
        file.addAction(a["add_root"])
        file.addSeparator()
        file.addAction(a["export"])
        file.addAction(a["screenshot"])
        file.addSeparator()
        file.addAction(a["quit"])
        self._rebuild_recent()

        view = bar.addMenu("&View")
        for dock in self.docks.values():
            view.addAction(dock.toggleViewAction())
        view.addSeparator()
        camera = view.addMenu("Camera")
        for name in ("cam_3d", "cam_bev", "cam_front", "cam_side", "cam_fit"):
            camera.addAction(a[name])
        view.addAction(a["rescale"])
        view.addAction(a["split"])
        view.addSeparator()
        view.addAction(a["reset_layout"])

        play = bar.addMenu("&Playback")
        for name in ("play", "prev", "next", "first", "last", "loop"):
            play.addAction(a[name])
        speed = play.addMenu("Speed")
        group = QtGui.QActionGroup(self)
        for label, value in SPEEDS:
            item = speed.addAction(label, lambda v=value: self.api.__setattr__("speed", v))
            item.setCheckable(True)
            item.setChecked(value == 1.0)
            group.addAction(item)
        play.addSeparator()
        play.addAction(a["cache_all"])

        filters = bar.addMenu("F&ilters")
        filters.addMenu(filter_menu(self, self.add_filter))
        filters.addSeparator()
        filters.addAction(a["load_preset"])
        filters.addAction(a["save_preset"])
        filters.addSeparator()
        filters.addAction(a["split"])
        filters.addAction(a["clear_pipeline"])

        python = bar.addMenu("P&ython")
        e = self.editor.actions_
        for name in ("new", "open", "save", "save_as"):
            python.addAction(e[name])
        python.addSeparator()
        python.addAction(e["run"])
        python.addAction(e["run_selection"])
        python.addAction(self.editor.auto_run)
        python.addMenu(self.editor.examples_menu)
        python.addSeparator()
        python.addAction(a["clear_pipeline"])
        python.addAction(a["clear_layers"])

        help_menu = bar.addMenu("&Help")
        for name in ("help_api", "help_keys", "about"):
            help_menu.addAction(a[name])

    def _make_toolbar(self) -> None:
        a = self.act
        bar = self.addToolBar("Main")
        bar.setObjectName("MainToolbar")
        bar.setIconSize(QtCore.QSize(18, 18))
        bar.addAction(a["open"])
        bar.addAction(a["export"])
        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel(" Color by "))
        self.toolbar_color = QtWidgets.QComboBox()
        self.toolbar_color.setMinimumWidth(120)
        self.toolbar_color.setModel(self.properties.color_by.model())  # same list as Properties
        self.toolbar_color.currentTextChanged.connect(
            lambda mode: mode and mode != self.display.color_by and self.set_color_by(mode))
        bar.addWidget(self.toolbar_color)
        bar.addAction(a["rescale"])
        bar.addSeparator()
        for name in ("cam_3d", "cam_bev", "cam_front", "cam_side"):
            bar.addAction(a[name])
        bar.addAction(a["cam_fit"])
        bar.addAction(a["split"])
        bar.addSeparator()
        bar.addAction(self.editor.actions_["run"])
        bar.addAction(a["cache_all"])

    def _make_status_bar(self) -> None:
        bar = self.statusBar()
        self.status_points = QtWidgets.QLabel()
        self.status_fps = QtWidgets.QLabel()
        self.status_cache = QtWidgets.QLabel()
        for label in (self.status_points, self.status_fps, self.status_cache):
            bar.addPermanentWidget(label)

    def _restore(self, restore_layout: bool) -> None:
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if restore_layout and geometry is not None:
            self.restoreGeometry(geometry)
        if not (restore_layout and state is not None and self.restoreState(state)):
            QtCore.QTimer.singleShot(0, self._default_sizes)
        self.editor.restore_session(self.settings)

    def reset_layout(self) -> None:
        for dock in self.docks.values():
            dock.setFloating(False)
            dock.show()
        self.restoreState(self._default_state)
        QtCore.QTimer.singleShot(0, self._default_sizes)

    # test runs

    def open_run_dialog(self) -> None:
        start = self.rec.path.parent if self.rec else data_dir()
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open test run (folder of .npz frames)", str(start))
        if path:
            self.open_run(path)

    def open_run(self, path) -> bool:
        path = str(Path(path).resolve())  # settings outlive the working directory
        try:
            rec = Recording(path)
            first = rec.get(0)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.warning(self, "Open test run", f"Could not open {path}:\n\n{e}")
            return False
        self.player.pause()
        if self.rec is not None:
            self.rec.close()
        self.rec = rec
        rec.on_loaded = self.frameLoaded.emit
        self.processor.set_recording(rec)
        self.index, self._wanted = 0, None
        self.raw = self.output = self.shown = None
        self._reported.clear()
        self.t0 = first.meta.get("timestamp")
        self.view.set_ground(estimate_ground(first))
        modes = color_modes(first)
        d = self.display
        d.clims.clear()  # value ranges differ between test runs
        if d.color_by not in modes:
            d.color_by = "intensity" if "intensity" in modes else "range"
        echoes = np.unique(first.echo).tolist() if "echo" in first.fields else []
        d.hidden_echoes &= set(echoes)
        self.properties.set_run(modes, echoes, rec.info.get("echoes", {}))
        self.player_bar.set_count(len(rec), rec.rate_hz)
        self.runs_panel.set_current(path)
        self._remember(path)
        self.shell.push(rec=rec)
        self.setWindowTitle(f"{rec.name} — LidarLab")
        self.pipeline_panel.rebuild()
        self.show_frame(0)
        self.view.set_camera("3d")
        self.statusBar().showMessage(
            f"Opened {rec.name}: {len(rec)} frames at {rec.rate_hz:.1f} Hz", 5000)
        return True

    def _remember(self, path: str) -> None:
        self.settings.setValue("lastRun", path)
        recent = [path] + [p for p in self._recent() if p != path]
        self.settings.setValue("recentRuns", recent[:MAX_RECENT])
        self._rebuild_recent()

    def _recent(self) -> list[str]:
        value = self.settings.value("recentRuns") or []
        return [value] if isinstance(value, str) else list(value)

    def _rebuild_recent(self) -> None:
        self.recent_menu.clear()
        for path in self._recent():
            self.recent_menu.addAction(path, lambda p=path: self.open_run(p))
        self.recent_menu.setEnabled(bool(self._recent()))

    def cache_all(self) -> None:
        if self.rec is not None:
            self.rec.cache_all(self.index)
            self.statusBar().showMessage("Loading all frames in the background…", 4000)

    # frames

    def show_frame(self, i: int, wait: bool = False) -> None:
        """Show frame ``i``; if it is not loaded yet, when it arrives (or now, with ``wait``)."""
        rec = self.rec
        if rec is None:
            return
        i = int(np.clip(i, 0, len(rec) - 1))
        self.index = i
        self.player_bar.set_index(i)
        raw = rec.get(i) if wait else rec.peek(i)
        if raw is None:
            if i in rec.errors:
                self.statusBar().showMessage(f"Frame {i} could not be read: {rec.errors[i]}", 8000)
                return
            self._wanted = i
            self.prefetch(i)
            return
        self._wanted = None
        self._display(raw)
        self.prefetch(i)

    def step(self, n: int) -> None:
        self.player.pause()
        self.show_frame(self.index + n)

    def refresh(self) -> None:
        if self.raw is not None:
            self._display(self.raw)

    def prefetch(self, start: int | None = None) -> None:
        """Queue the frames the playhead needs next (replaces the previous request)."""
        rec = self.rec
        if rec is None:
            return
        n = len(rec)
        i = self.index if start is None else start
        if self.player.playing:
            speed = self.player.speed
            ahead = 30 if speed <= 0 else int(np.clip(2.0 * rec.rate_hz * speed, 10, 60))
            order = [i + k for k in range(ahead)]
            if self.player.loop:
                order = [j % n for j in order]
        else:
            order = [i, i + 1, i - 1, i + 2, i + 3, i - 2, i + 4, i + 5]
        order = [j for j in order if 0 <= j < n]
        rec.prefetch(order)
        self.processor.request(order)

    def frame_ready(self, i: int) -> bool:
        """Frame ``i`` can be shown without waiting: loaded and, with steps, processed."""
        if self.rec is None or not self.rec.is_cached(i):
            return False
        return not self.pipeline.active or self.processor.has(i)

    def _frame_loaded(self, i: int) -> None:
        self.processor.notify()
        if self._wanted == i and self.rec is not None:
            self.show_frame(i)

    def _frame_processed(self, i: int) -> None:
        if self.preview_pending == i and self.raw is not None and self.index == i:
            self.preview_pending = None
            self._display(self.raw)

    def apply_live(self, change) -> None:
        """Run ``change()`` (a parameter edit) and show the result when the background
        threads have it, so a slow filter does not block the window while a slider moves."""
        self._live = True
        try:
            change()
        finally:
            self._live = False

    def _process(self, raw: Frame) -> tuple[Frame, float]:
        """Pipeline output for ``raw``: from the background cache, or computed now."""
        if not self.pipeline.active:
            return raw, 0.0
        i = raw.meta.get("frame", self.index)
        result = self.processor.get(i)
        if result is None:
            result = self.processor.compute(i, raw)
        self.pipeline.record(result.stats)
        return result.frame, result.ms

    def _display(self, raw: Frame) -> None:
        start = time.perf_counter()
        self.preview_pending = None  # this draw shows the current pipeline version
        self.raw = raw
        self.output, steps_ms = self._process(raw)
        lookup = time.perf_counter()
        self._update_shown()
        self._draw()
        self.timing = {"steps": steps_ms, "get output": (lookup - start) * 1e3,
                       "draw": (time.perf_counter() - lookup) * 1e3}
        self.shell.push(frame=raw)
        self.fps.tick()
        self._report_errors()
        now = time.perf_counter()
        if not self.player.playing or now - self._last_info > 0.25:
            self._last_info = now
            self.info.show_frame(self)
            self.pipeline_panel.update_stats()
            self.status_points.setText(f"{len(self.shown):,} points")

    def _visible(self, frame: Frame | None) -> Frame | None:
        """``frame`` without the echoes hidden under Properties."""
        hidden = self.display.hidden_echoes
        if frame is not None and hidden and "echo" in frame.fields:
            return frame[~np.isin(frame.echo, list(hidden))]
        return frame

    def _update_shown(self) -> None:
        self.shown = self._visible(self.output)

    def _draw(self) -> None:
        shown = self.shown
        if shown is None:
            return
        d = self.display
        mode = d.color_by
        values = None if mode == SOLID else color_values(shown, mode)
        lut = categories = None
        if values is not None and mode == "echo":
            echoes = sorted(set(np.unique(values).tolist()) | set(self.properties.echo_checks))
            labels = tuple(f"echo {e}" for e in echoes)
            lut = category_lut(tuple(echoes), labels)
            categories = [(label, category_color(e))
                          for e, label in zip(echoes, labels, strict=True)
                          if e not in d.hidden_echoes]
        elif values is not None and mode in CATEGORICAL:
            classes = np.unique(values).tolist()
            names = self.rec.info.get("labels", {}) if self.rec is not None else {}
            labels = tuple(f"{c} {names[str(c)]}" if str(c) in names else str(c) for c in classes)
            lut = category_lut(tuple(classes), labels, LABEL_COLORS)
            categories = [(label, category_color(c, LABEL_COLORS))
                          for c, label in zip(classes, labels, strict=True)]
        elif values is not None:
            if mode not in d.clims:
                d.clims[mode] = auto_range(values)
                self.properties.sync()
            lut = colormap_lut(d.cmap(mode), *d.clims[mode])
        self.view.set_cloud(shown.xyz, values, lut, mode, legend=d.legend, categories=categories)
        if self.view.split and self.raw is not None:
            raw = self._visible(self.raw)
            raw_values = None if values is None else color_values(raw, mode)
            self.view.set_raw_cloud(raw.xyz, raw_values, lut)
        if d.show_removed and self.raw is not None and len(self.output) < len(self.raw):
            removed = removed_points(self.raw, self.output)
            if d.hidden_echoes and "echo" in removed.fields:
                removed = removed[~np.isin(removed.echo, list(d.hidden_echoes))]
            self.view.set_layer(REMOVED, removed.xyz, "#ff3b3b", d.point_size + 1)
        else:
            self.view.set_layer(REMOVED, None)
        for layer in self.layers.values():
            self._draw_layer(layer)
        self.view.render()

    def _draw_layer(self, layer: Layer) -> None:
        xyz = None
        if layer.visible and layer.func is not None and self.shown is not None:
            try:
                result = layer.func(self.shown)
                if isinstance(result, Frame):
                    xyz = result.xyz
                else:
                    arr = np.asarray(result)
                    if arr.dtype == bool and arr.shape == (len(self.shown),):
                        xyz = self.shown.xyz[arr]
                    elif arr.ndim == 2 and arr.shape[1] == 3:
                        xyz = arr
                    else:
                        raise TypeError(f"must return a boolean mask of {len(self.shown)} values, "
                                        f"a Frame or (N, 3) points; got shape {arr.shape}")
                layer.error = None
            except Exception as e:  # noqa: BLE001 - reported once in the shell
                layer.error = e
        elif layer.visible:
            xyz = layer.points
        layer.count = 0 if xyz is None else len(xyz)
        self.view.set_layer(layer.name, xyz, layer.color, layer.size)

    def _report_errors(self) -> None:
        """Print each new step or highlight error once (they fail on every frame)."""
        failed = [(f"step '{s.name}'", s.error) for s in self.pipeline]
        failed += [(f"highlight '{layer.name}'", layer.error) for layer in self.layers.values()]
        for label, error in failed:
            text = None if error is None else f"{type(error).__name__}: {error}"
            if text is not None and self._reported.get(label) != text:
                trace = "".join(traceback.format_exception(error))
                self.shell.print_error(
                    f"{label} failed on frame {self.index} and is skipped:\n{trace}")
                self.statusBar().showMessage(f"{label} failed: {text}", 8000)
            self._reported[label] = text

    # display settings

    def apply_display(self) -> None:
        """Redraw after a change to ``self.display``."""
        d = self.display
        self.view.set_split(d.split)
        self.act["split"].setChecked(d.split)
        self.view.set_point_size(d.point_size)
        self.view.set_background(d.background)
        self.view.show_helpers(rings=d.rings, grid=d.grid, ego=d.ego, axes=d.axes)
        self._update_shown()
        if self.shown is not None:
            self._draw()
            self.info.show_frame(self)
            self.status_points.setText(f"{len(self.shown):,} points")
        else:
            self.view.render()
        self.properties.sync()
        if self.toolbar_color.currentText() != d.color_by:
            self.toolbar_color.setCurrentText(d.color_by)

    def set_color_by(self, mode: str) -> None:
        self.display.color_by = mode
        self.apply_display()

    def rescale(self) -> None:
        self.display.clims.pop(self.display.color_by, None)
        self.apply_display()

    def _cycle_color(self) -> None:
        modes = self.api.color_modes
        if modes:
            k = modes.index(self.display.color_by) if self.display.color_by in modes else -1
            self.set_color_by(modes[(k + 1) % len(modes)])

    def set_camera(self, preset: str) -> None:
        self.view.set_camera(preset)

    def set_cloud_visible(self, visible: bool) -> None:
        self.view.cloud.visible = visible
        self.view.render()

    def toggle_split(self, on: bool | None = None) -> None:
        self.display.split = (not self.display.split) if on is None else bool(on)
        self.apply_display()

    # filters and presets

    def add_filter(self, key: str, name: str | None = None, values: dict | None = None) -> str:
        """Add built-in filter ``key`` after the last step and select it; returns its name."""
        flt = available()[key](**(values or {}))
        name = self.pipeline.unique_name(name or flt.name)
        self.pipeline.add(flt, name=name)
        self.pipeline_panel.select_step(name)
        self.docks["properties"].raise_()
        return name

    def _preset_folder(self) -> Path:
        saved = self.settings.value("presetDir")
        if saved and Path(saved).is_dir():
            return Path(saved)
        folder = scripts_dir() / "presets"
        return folder if folder.is_dir() else examples_dir() / "presets"

    def load_preset(self, path: str | None = None) -> bool:
        path = path or preset_dialog(self, False, str(self._preset_folder()))
        if not path:
            return False
        try:
            self.pipeline.load(path, run_script=self.shell.run_file_now)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.warning(self, "Load pipeline preset", f"{path}:\n\n{e}")
            return False
        self.settings.setValue("presetDir", str(Path(path).parent))
        self.statusBar().showMessage(f"Loaded {len(self.pipeline)} steps from {path}", 6000)
        if len(self.pipeline):
            self.pipeline_panel.select_step(self.pipeline[-1].name)
        return True

    def save_preset(self, path: str | None = None) -> bool:
        if not len(self.pipeline):
            self.statusBar().showMessage("No steps to save", 4000)
            return False
        if path is None:
            folder = self._preset_folder()
            if folder == examples_dir() / "presets":
                folder = scripts_dir() / "presets"  # examples ship with the package
                folder.mkdir(parents=True, exist_ok=True)
            path = preset_dialog(self, True, str(folder / "pipeline.yaml"))
        if not path:
            return False
        try:
            self.pipeline.save(path)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.warning(self, "Save pipeline preset", str(e))
            return False
        self.settings.setValue("presetDir", str(Path(path).parent))
        self.statusBar().showMessage(f"Saved {len(self.pipeline)} steps to {path}", 6000)
        return True

    # layers

    def add_layer(self, layer: Layer) -> None:
        self.layers[layer.name] = layer
        self.pipeline_panel.rebuild()
        if self.shown is not None:
            self._draw_layer(layer)
            self.view.render()
            self._report_errors()
            self.pipeline_panel.update_stats()

    def remove_layer(self, name: str) -> None:
        self.layers.pop(name, None)
        self.view.set_layer(name, None)
        self.view.render()
        self.pipeline_panel.rebuild()

    def clear_layers(self) -> None:
        for name in list(self.layers):
            self.view.set_layer(name, None)
        self.layers.clear()
        self.view.render()
        self.pipeline_panel.rebuild()

    def set_layer_visible(self, name: str, visible: bool) -> None:
        layer = self.layers.get(name)
        if layer is not None:
            layer.visible = visible
            self._draw_layer(layer)
            self.view.render()

    # events

    def _pipeline_changed(self) -> None:
        self.processor.invalidate()
        self.pipeline_panel.rebuild()
        self.properties.step_editor.sync()
        if self._live and self.pipeline.active and self.raw is not None:
            self.preview_pending = self.index  # shown by _frame_processed
            self.properties.step_editor.stats.setText("computing…")
        else:
            self.preview_pending = None
            self.refresh()  # scripts expect view.output to be current right away
        self.prefetch()  # (re)compute the frames around the playhead, current one first

    def _play_state_changed(self, playing: bool) -> None:
        self.act["play"].setIcon(theme.icon("pause" if playing else "play"))
        if not playing and self.raw is not None:
            self.info.show_frame(self)
            self.pipeline_panel.update_stats()

    def _poll_status(self) -> None:
        rec = self.rec
        if rec is None:
            return
        if rec.cache_version != self._cache_version:
            self._cache_version = rec.cache_version
            mask = rec.cached_mask()
            self.player_bar.seekbar.set_cached(mask)
            self.status_cache.setText(f"memory: {mask.sum()}/{len(rec)} frames, "
                                      f"{rec.cached_bytes / 2**30:.2f} GB")
        if self.processor.version != self._processed_version:
            self._processed_version = self.processor.version
            mask = np.zeros(len(rec), bool)
            if self.pipeline.active:
                mask[[i for i in self.processor.cached() if i < len(rec)]] = True
            self.player_bar.seekbar.set_processed(mask)
        self.status_fps.setText(f"{self.fps.value:4.1f} fps" if self.player.playing else "paused")

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        mine = isinstance(obj, QtWidgets.QWidget) and obj.window() is self
        if kind == QtCore.QEvent.Type.KeyPress and mine:
            if not isinstance(obj, TEXT_INPUTS) and self._player_key(event):
                return True
            if self.view.owns(obj):
                return True  # VTK's own keys stay off (q quits, 3 switches to stereo, …)
        elif kind == QtCore.QEvent.Type.KeyRelease and self.view.owns(obj):
            return True
        return super().eventFilter(obj, event)

    def _player_key(self, event: QtGui.QKeyEvent) -> bool:
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return False
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        key, k = event.key(), Qt.Key
        cameras = {k.Key_1: "3d", k.Key_2: "bev", k.Key_3: "front", k.Key_4: "side", k.Key_R: "fit"}
        if key == k.Key_Space:
            self.player.toggle()
        elif key in (k.Key_Right, k.Key_Left):
            self.step((10 if shift else 1) * (1 if key == k.Key_Right else -1))
        elif key == k.Key_Home:
            self.show_frame(0)
        elif key == k.Key_End and self.rec is not None:
            self.show_frame(len(self.rec) - 1)
        elif key in cameras:
            self.set_camera(cameras[key])
        elif key == k.Key_C:
            self._cycle_color()
        elif key in (k.Key_Plus, k.Key_Equal, k.Key_Minus):
            change = -0.5 if key == k.Key_Minus else 0.5
            self.display.point_size = float(np.clip(self.display.point_size + change, 1.0, 10.0))
            self.apply_display()
        else:
            return False
        return True

    def _report_exception(self, kind, value, tb) -> None:
        text = "".join(traceback.format_exception(kind, value, tb))
        self.shell.print_error(text)
        self.statusBar().showMessage(f"{kind.__name__}: {value}", 8000)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self.editor.maybe_save_all():
            event.ignore()
            return
        self.player.pause()
        self._poll.stop()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.editor.save_session(self.settings)
        QtWidgets.QApplication.instance().removeEventFilter(self)
        sys.excepthook = self._excepthook
        self.pipeline.unsubscribe(self._pipeline_changed)
        # threads finishing a frame after this must not signal the deleted window
        self.processor.on_done = None
        self.processor.close()
        if self.rec is not None:
            self.rec.on_loaded = None
            self.rec.close()
        self.shell.shutdown()
        self.view.close()
        super().closeEvent(event)

    # output

    def export_frame(self) -> None:
        if self.output is None:
            return
        folder = Path(self.settings.value("exportDir") or data_dir() / "exports" / self.rec.name)
        suggested = folder / f"frame_{self.index:06d}.npz"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export frame", str(suggested),
                                                        "NumPy archive (*.npz)")
        if path:
            write_frame(self.output, path)
            self.settings.setValue("exportDir", str(Path(path).parent))
            self.statusBar().showMessage(f"Saved {len(self.output):,} points to {path}", 6000)

    def save_screenshot(self) -> None:
        name = f"{self.rec.name}_frame{self.index:06d}.png" if self.rec else "view.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save view screenshot",
                                                        str(Path.cwd() / name), "PNG image (*.png)")
        if path:
            self.view.save_screenshot(path)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def grab_window(self) -> QtGui.QImage:
        """The whole window as an image, with the 3D view pasted in (grab() cannot see OpenGL)."""
        pixmap = self.grab()
        painter = QtGui.QPainter(pixmap)
        top_left = self.view.mapTo(self, QtCore.QPoint(0, 0))
        painter.drawImage(QtCore.QRect(top_left, self.view.size()), self.view.grab_image())
        painter.end()
        return pixmap.toImage()

    # help

    def _show_html(self, title: str, html: str) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 620)
        text = QtWidgets.QTextBrowser()
        text.setHtml(html)
        text.setOpenExternalLinks(True)
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(text)
        dialog.show()

    def show_scripting_help(self) -> None:
        self._show_html("Scripting reference", HELP_HTML)

    def show_keys_help(self) -> None:
        self._show_html("Keyboard and mouse", KEYS_HTML)

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, "About LidarLab",
            f"<b>LidarLab {lidarlab.__version__}</b><br>LiDAR test run viewer with an integrated "
            "Python shell and editor.<br><br>Qt (PySide6) · VTK · IPython · NumPy")
