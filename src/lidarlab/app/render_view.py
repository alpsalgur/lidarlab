"""3D view: point layers, color legend, scene helpers and camera presets."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyvista as pv
from matplotlib import colormaps
from PySide6 import QtGui, QtWidgets
from pyvistaqt import QtInteractor
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints, vtkVariant
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkTextActor

ECHO_COLORS = ("#ff9f1c", "#d9d9d9", "#4dabf7", "#69db7c", "#da77f2", "#ffd43b")
# ground-truth classes; the synthetic run's 0-6 are cover, ground, object, edge, vegetation,
# rain, spray. Other ids (e.g. WADS snow 110) take a color by value.
LABEL_COLORS = ("#ff9f1c", "#8d6e63", "#b0bec5", "#4db6ac", "#66bb6a", "#4fc3f7", "#e040fb",
                "#ffd54f", "#ef5350", "#7e57c2", "#26c6da", "#d4e157")
BACKGROUNDS = {
    "Dark gradient": ("#0b0d12", "#2b2f3d"),
    "Black": ("#000000", None),
    "Blue-gray": ("#52576e", None),
    "White": ("#ffffff", None),
}
# sensor frame: x forward, y left, z up
CAMERAS = {
    "3d": {"position": (-30.0, -18.0, 22.0), "focal": (14.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)},
    "bev": {"position": (15.0, 0.0, 170.0), "focal": (15.0, 0.0, 0.0), "up": (1.0, 0.0, 0.0),
            "scale": 42.0},
    "front": {"position": (-14.0, 0.0, 5.0), "focal": (35.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)},
    "side": {"position": (12.0, -75.0, 12.0), "focal": (12.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)},
}
RING_RADII = (10, 20, 30, 40, 50, 60, 80, 100)
EGO_SIZE = (4.7, 1.9, 1.5)  # length, width, height of the car outline, m
LEGEND_BACKGROUND = (0.07, 0.08, 0.1)


def _plain_text(prop, size: int) -> None:
    prop.SetFontSize(size)
    prop.SetBold(False)
    prop.SetItalic(False)
    prop.SetShadow(False)
    prop.SetColor(0.85, 0.85, 0.85)


def _corner_label(text: str) -> vtkTextActor:
    actor = vtkTextActor()
    actor.SetInput(f" {text} ")
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.012, 0.955)
    prop = actor.GetTextProperty()
    _plain_text(prop, 14)
    prop.SetVerticalJustificationToTop()
    prop.SetBackgroundColor(*LEGEND_BACKGROUND)
    prop.SetBackgroundOpacity(0.7)
    return actor


_ids = np.arange(1, dtype=np.int64)


def _vertex_ids(n: int) -> np.ndarray:
    """0..n as vtkIdType, shared between frames so vertex cells need no new array."""
    global _ids
    if len(_ids) < n + 1:
        _ids = np.arange(max(n + 1, 2 * len(_ids)), dtype=np.int64)
    return _ids


def points_polydata(xyz: np.ndarray, scalars: np.ndarray | None = None) -> vtkPolyData:
    """Vertex polydata that reads the numpy buffers directly (no copy)."""
    n = len(xyz)
    ids = _vertex_ids(n)
    points = vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(xyz, deep=False))
    cells = vtkCellArray()
    cells.SetData(numpy_support.numpy_to_vtkIdTypeArray(ids[: n + 1], deep=False),
                  numpy_support.numpy_to_vtkIdTypeArray(ids[:n], deep=False))
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(cells)
    if scalars is not None:
        array = numpy_support.numpy_to_vtk(scalars, deep=False)
        array.SetName("scalars")
        poly.GetPointData().SetScalars(array)
    return poly


@lru_cache(maxsize=64)
def colormap_lut(name: str, lo: float, hi: float) -> vtkLookupTable:
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.Build()
    for i, rgba in enumerate(colormaps[name](np.linspace(0.0, 1.0, 256))):
        lut.SetTableValue(i, *rgba)
    lut.SetTableRange(lo, hi if hi > lo else lo + 1e-6)
    lut.SetNanColor(0.5, 0.5, 0.5, 1.0)
    return lut


def category_color(value: int, palette: tuple[str, ...] = ECHO_COLORS) -> str:
    return palette[int(value) % len(palette)]


@lru_cache(maxsize=16)
def category_lut(values: tuple[int, ...], labels: tuple[str, ...],
                 palette: tuple[str, ...] = ECHO_COLORS) -> vtkLookupTable:
    """One color per value (echo numbers, classes), with labels for the legend."""
    lut = vtkLookupTable()
    lut.IndexedLookupOn()
    lut.SetNumberOfTableValues(len(values))
    for k, (value, label) in enumerate(zip(values, labels, strict=True)):
        lut.SetTableValue(k, *pv.Color(category_color(value, palette)).float_rgba)
        lut.SetAnnotation(vtkVariant(int(value)), label)
    lut.SetNanColor(0.5, 0.5, 0.5, 1.0)
    return lut


class PointLayer:
    """One actor drawing points, colored through a lookup table or in a single color."""

    def __init__(self, renderer, size: float = 2.0, color: str = "#ffffff", round_points=False):
        self.renderer = renderer
        self.mapper = vtkPolyDataMapper()
        self.mapper.SetColorModeToMapScalars()
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.UseLookupTableScalarRangeOn()
        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        prop = self.actor.GetProperty()
        prop.LightingOff()
        prop.SetRenderPointsAsSpheres(round_points)
        self.set_size(size)
        self.set_color(color)
        renderer.AddActor(self.actor)
        self.count = 0
        self._arrays = None

    def set_points(self, xyz: np.ndarray, scalars=None, lut=None) -> None:
        self._arrays = (xyz, scalars)  # VTK reads these buffers; keep them alive
        self.mapper.SetInputData(points_polydata(xyz, scalars))
        self.mapper.SetScalarVisibility(scalars is not None)
        if lut is not None:
            self.mapper.SetLookupTable(lut)
        self.count = len(xyz)

    def set_size(self, size: float) -> None:
        self.actor.GetProperty().SetPointSize(size)

    def set_color(self, color: str) -> None:
        self.actor.GetProperty().SetColor(pv.Color(color).float_rgb)

    @property
    def visible(self) -> bool:
        return bool(self.actor.GetVisibility())

    @visible.setter
    def visible(self, on: bool) -> None:
        self.actor.SetVisibility(on)

    def remove(self) -> None:
        self.renderer.RemoveActor(self.actor)


def ring_mesh(z: float, segments: int = 240) -> pv.PolyData:
    t = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    points, lines = [], []
    for k, r in enumerate(RING_RADII):
        points.append(np.c_[r * np.cos(t), r * np.sin(t), np.full_like(t, z)])
        lines.append(np.r_[len(t), np.arange(len(t)) + k * len(t)])
    return pv.PolyData(np.vstack(points).astype(np.float32), lines=np.concatenate(lines))


class RenderView(QtWidgets.QWidget):
    """The 3D view: the point cloud, named point layers, color legend and scene helpers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = QtInteractor(self, auto_update=False, multi_samples=4)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)
        self.renderer = self.plotter.renderer
        self.cloud = PointLayer(self.renderer, size=2.0)
        self.raw_renderer: vtkRenderer | None = None  # left half in split view
        self.raw_cloud: PointLayer | None = None
        self._split_props: list = []
        self._raw_helpers: list = []
        self.layers: dict[str, PointLayer] = {}
        self._text_color = (0.85, 0.85, 0.85)
        self._category_actors: list[vtkTextActor] = []
        self._category_key = None
        self.legend = self._make_legend()
        self._helpers: dict[str, object] = {}
        self._helper_flags = {"rings": True, "grid": False, "ego": True}
        self._ground: float | None = None
        self.plotter.add_axes(line_width=2, color="#d4d4d4")
        self.set_background("Dark gradient")
        self.set_ground(-1.8)
        self.set_camera("3d")

    def owns(self, obj) -> bool:
        """True for the VTK widget and its children (their key events are filtered)."""
        return isinstance(obj, QtWidgets.QWidget) and (obj is self or self.isAncestorOf(obj))

    def _make_legend(self) -> vtkScalarBarActor:
        """Vertical color bar on the right with a dark background."""
        bar = vtkScalarBarActor()
        bar.SetOrientationToVertical()
        bar.SetPosition(0.875, 0.2)
        bar.SetWidth(0.11)
        bar.SetHeight(0.6)
        bar.SetBarRatio(0.22)
        bar.SetTitleRatio(0.4)
        bar.SetNumberOfLabels(5)
        bar.SetLabelFormat("%.3g")
        bar.SetVerticalTitleSeparation(6)
        bar.SetTextPad(3)
        bar.UnconstrainedFontSizeOn()
        bar.DrawBackgroundOn()
        bar.GetBackgroundProperty().SetColor(*LEGEND_BACKGROUND)
        bar.GetBackgroundProperty().SetOpacity(0.7)
        _plain_text(bar.GetTitleTextProperty(), 14)
        _plain_text(bar.GetLabelTextProperty(), 13)
        bar.SetVisibility(False)
        self.renderer.AddViewProp(bar)
        return bar

    def _set_categories(self, title: str, items: list[tuple[str, str]]) -> None:
        """Legend for categories (echoes): one label per value, in its color.

        VTK's scalar bar puts category labels outside its own box, over the points."""
        for actor in self._category_actors:
            self.renderer.RemoveViewProp(actor)
        self._category_actors = []
        if not items:
            return
        if len(items) > 13:  # keep the list on screen
            items = [*items[:12], (f"… {len(items) - 12} more", "#8b949e")]
        for k, (label, color) in enumerate([(title, None), *items]):
            actor = vtkTextActor()
            actor.SetInput(f" {label} ")
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            actor.SetPosition(0.985, 0.78 - 0.05 * k)
            prop = actor.GetTextProperty()
            _plain_text(prop, 14 if color is None else 13)
            prop.SetJustificationToRight()
            prop.SetColor(pv.Color(color).float_rgb if color else self._text_color)
            prop.SetBackgroundColor(*LEGEND_BACKGROUND)
            prop.SetBackgroundOpacity(0.7)
            self.renderer.AddViewProp(actor)
            self._category_actors.append(actor)

    # content

    def set_cloud(self, xyz, scalars=None, lut=None, title: str = "", legend: bool = True,
                  categories: list[tuple[str, str]] | None = None) -> None:
        """Draw the cloud. ``categories`` ((label, color) per value) replaces the color bar."""
        self.cloud.set_points(xyz, scalars, lut)
        show = legend and lut is not None and scalars is not None
        bar = show and not categories
        if bar:
            self.legend.SetLookupTable(lut)
            self.legend.SetTitle(title)
        self.legend.SetVisibility(bar)
        key = (title, tuple(categories)) if show and categories else None
        if key != self._category_key:
            self._category_key = key
            self._set_categories(title, list(categories) if key else [])

    def set_layer(self, name: str, xyz, color: str = "#ff3b3b", size: float = 4.0) -> None:
        """Draw ``xyz`` as layer ``name``; None removes the layer."""
        layer = self.layers.get(name)
        if xyz is None:
            if layer is not None:
                layer.remove()
                del self.layers[name]
            return
        if layer is None:
            layer = self.layers[name] = PointLayer(self.renderer, round_points=True)
        layer.set_color(color)
        layer.set_size(size)
        layer.set_points(np.ascontiguousarray(xyz, dtype=np.float32))

    def set_point_size(self, size: float) -> None:
        self.cloud.set_size(size)
        if self.raw_cloud is not None:
            self.raw_cloud.set_size(size)

    # split view

    @property
    def split(self) -> bool:
        return self.raw_renderer is not None

    def set_split(self, on: bool) -> None:
        """Raw points in the left half, the main cloud in the right; one shared camera."""
        if on == self.split:
            return
        window = self.plotter.render_window
        if on:
            r = vtkRenderer()
            r.SetViewport(0.0, 0.0, 0.5, 1.0)
            r.SetActiveCamera(self.renderer.GetActiveCamera())  # both halves move together
            self.raw_renderer = r
            self.raw_cloud = PointLayer(r, size=self.cloud.actor.GetProperty().GetPointSize())
            self._copy_background()
            self._share_helpers()
            r.AddViewProp(_corner_label("raw"))
            main_label = _corner_label("after pipeline")
            self.renderer.AddViewProp(main_label)
            self._split_props = [main_label]
            window.AddRenderer(r)
            self.renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
        else:
            window.RemoveRenderer(self.raw_renderer)
            for prop in self._split_props:
                self.renderer.RemoveViewProp(prop)
            self.raw_renderer = self.raw_cloud = None
            self._split_props = []
            self.renderer.SetViewport(0.0, 0.0, 1.0, 1.0)
        self.render()

    def set_raw_cloud(self, xyz, scalars=None, lut=None) -> None:
        if self.raw_cloud is not None:
            self.raw_cloud.set_points(xyz, scalars, lut)

    def _copy_background(self) -> None:
        if self.raw_renderer is not None:
            r, main = self.raw_renderer, self.renderer
            r.SetBackground(main.GetBackground())
            r.SetBackground2(main.GetBackground2())
            r.SetGradientBackground(main.GetGradientBackground())

    def _share_helpers(self) -> None:
        """Show the main view's rings, grid and ego car in the raw half too."""
        if self.raw_renderer is None:
            return
        for actor in self._raw_helpers:
            self.raw_renderer.RemoveViewProp(actor)
        self._raw_helpers = list(self._helpers.values())
        for actor in self._raw_helpers:
            self.raw_renderer.AddViewProp(actor)

    # scene

    def set_background(self, name: str) -> None:
        bottom, top = BACKGROUNDS.get(name, BACKGROUNDS["Dark gradient"])
        self.plotter.set_background(bottom, top=top)
        # legend text sits on its own dark background, so it stays light; helpers adapt
        self._text_color = (0.85, 0.85, 0.85)
        self._copy_background()

    def set_ground(self, z: float) -> None:
        """Rebuild the range rings, grid and ego car outline at ground height ``z``."""
        if self._ground is not None and abs(z - self._ground) < 0.05:
            return
        self._ground = z
        add = dict(lighting=False, pickable=False, render=False, reset_camera=False)
        self._helpers["rings"] = self.plotter.add_mesh(ring_mesh(z), color="#3d4454", line_width=1,
                                                       name="rings", **add)
        labels = np.array([[r * np.cos(0.6), r * np.sin(0.6), z] for r in (20, 40, 60, 80, 100)])
        self._helpers["ring_labels"] = self.plotter.add_point_labels(
            labels, [f"{r} m" for r in (20, 40, 60, 80, 100)], font_size=11, text_color="#7b8394",
            shape=None, show_points=False, always_visible=True, name="ring_labels", render=False,
            reset_camera=False)
        grid = pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=200, j_size=200,
                        i_resolution=20, j_resolution=20)
        self._helpers["grid"] = self.plotter.add_mesh(grid, style="wireframe", color="#2a2f3a",
                                                      name="grid", **add)
        length, width, height = EGO_SIZE
        ego = pv.Box(bounds=(-length / 2, length / 2, -width / 2, width / 2, z, z + height))
        self._helpers["ego"] = self.plotter.add_mesh(ego, style="wireframe", color="#4fc3f7",
                                                     line_width=2, name="ego", **add)
        self.show_helpers(**self._helper_flags)
        self._share_helpers()

    def show_helpers(self, rings: bool = True, grid: bool = False, ego: bool = True,
                     axes: bool | None = None) -> None:
        self._helper_flags = {"rings": rings, "grid": grid, "ego": ego}
        visible = {"rings": rings, "ring_labels": rings, "grid": grid, "ego": ego}
        for name, actor in self._helpers.items():
            actor.SetVisibility(visible[name])
        if axes is not None:
            if axes:
                self.plotter.show_axes()
            else:
                self.plotter.hide_axes()

    def set_camera(self, preset: str) -> None:
        camera = self.renderer.GetActiveCamera()
        if preset == "fit":
            bounds = self.cloud.actor.GetBounds() if self.cloud.count else None
            self.renderer.ResetCamera(*([bounds] if bounds else []))
        else:
            p = CAMERAS[preset]
            camera.SetParallelProjection(preset == "bev")
            camera.SetPosition(*p["position"])
            camera.SetFocalPoint(*p["focal"])
            camera.SetViewUp(*p["up"])
            if "scale" in p:
                camera.SetParallelScale(p["scale"])
        self.render()

    def render(self) -> None:
        self.renderer.ResetCameraClippingRange()
        if self.raw_renderer is not None:
            self.raw_renderer.ResetCameraClippingRange()
        self.plotter.render()

    # output

    def grab_image(self) -> QtGui.QImage:
        """The rendered 3D image (QWidget.grab cannot see the OpenGL surface)."""
        img = np.ascontiguousarray(self.plotter.screenshot(return_img=True)[..., :3])
        h, w, _ = img.shape
        return QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()

    def save_screenshot(self, path) -> None:
        self.plotter.screenshot(str(path))

    def close(self) -> None:
        self.plotter.close()
        super().close()
