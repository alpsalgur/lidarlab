"""The ``view`` object of the Python shell: drives the viewer from scripts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from lidarlab import Frame

from .player import Layer

HELP_HTML = """
<h3>LidarLab scripting</h3>
<p>The Python shell and the script editor share one namespace with the viewer:</p>
<table cellpadding="3">
<tr><td><b>rec</b></td><td>current test run: <code>rec[42]</code>, <code>len(rec)</code>,
  <code>rec.rate_hz</code>, <code>rec.info</code></td></tr>
<tr><td><b>frame</b></td><td>current frame as loaded (updated on every frame change)</td></tr>
<tr><td><b>view</b></td><td>the viewer (below)</td></tr>
<tr><td><b>pipe</b></td><td>processing steps applied to every frame</td></tr>
<tr><td><b>lab</b>, <b>np</b></td><td>the lidarlab module, NumPy</td></tr>
</table>
<h4>Frames</h4>
<pre>frame.xyz                  (N, 3) float32, sensor frame: x forward, y left, z up
frame.intensity, frame.echo, frame.fields     per-point fields
frame.range, .azimuth, .elevation, .pulse     derived (m, deg, pulse id)
frame[frame.echo != 0]     subset (all fields, keeps the source index)
frame["new"] = values      add or replace a field;  frame.copy() for writable arrays
frame.meta                 timestamp, roll, pitch, yaw, frame number, path
frame.save("out.npz")      write in the source layout</pre>
<h4>Processing steps (applied while playing)</h4>
<pre>@lab.step
def remove_cover_echo(f):
    return f.echo != 0          # keep-mask, a new Frame, or None after f["x"] = ...
pipe                            # list with points in/out and time per step
pipe.enable("remove_cover_echo", False);  pipe.remove(...);  pipe.clear()
view.refresh()                  # re-run after changing a variable a step reads</pre>
<h4>Filters with parameters (sliders under Properties)</h4>
<pre>pipe.add(lab.filters.DROR(beta=3))          # built-in: lab.filters.available()
pipe["dror"].set(min_neighbors=3);  pipe["dror"].values;  pipe["dror"].reset()

@lab.filter(radius=(0.05, 2.0), k=(1, 20))  # ranges make sliders
def my_ror(f, radius=0.4, k=2):             # defaults come from the signature
    ...
pipe.save("denoise.yaml");  pipe.load("denoise.yaml")     # presets</pre>
<p>Steps run ahead of the playhead in background threads; the green line under the seek
bar shows the frames already processed.</p>
<h4>Viewer</h4>
<pre>view.goto(120); view.next(); view.prev(); view.play(); view.pause(); view.speed = 2
view.index, view.frame, view.output (after the steps), view.shown (what is drawn)
view.color_by("intensity", cmap="turbo", clim=(0, 100))
view.point_size = 3;  view.camera("bev");  view.echoes = {1, 2};  view.show_removed()
view.split()                    # raw left, after the pipeline right; view.split(False)
job = view.export("data/exports/clean", frames=range(100, 200))   # in the background
job.done, job.report.summary()  # lab.export_run(rec, "out", pipe) does the same, blocking</pre>
<h4>Analysis</h4>
<pre>view.tool("pick")               # also "measure", "select", "navigate" (keys P, M, S, N; Esc)
view.picked                     # all values of the picked point
view.measurement                # distance, horizontal, dz of the last two clicked points
view.selection                  # points of view.shown inside the box (S, then drag)
view.selection.intensity.mean() # the box stays fixed to the sensor: same region every frame
view.bookmark("spray starts");  view.bookmarks      # B adds one, [ and ] jump
view.metrics["removed_pct"]     # one value per frame (NaN until measured); view.compute_metrics()
view.save_workspace("analysis.lidarlab");  view.open_workspace("analysis.lidarlab")
@view.highlight(color="red")    # paint points on every frame, without removing them
def weak(f): return f.intensity < 10
view.show(points, name="mine", color="yellow")   # fixed points: Frame, (N, 3) or mask
view.remove("mine");  view.clear_layers();  view.screenshot("shot.png")</pre>
"""


class ViewAPI:
    """Controls the viewer from Python. See F1 (Help > Scripting Reference).

        view.goto(120)
        view.color_by("intensity", cmap="turbo", clim=(0, 120))
        view.output                         # current frame after the processing steps

        @view.highlight(color="red")        # paint matching points on every frame
        def weak_near(f):
            return (f.intensity < 10) & (f.range < 20)
    """

    def __init__(self, window):
        self._w = window

    def __repr__(self) -> str:
        w = self._w
        if w.rec is None:
            return "<view: no test run open>"
        return (f"<view: {w.rec.name}, frame {w.index}/{len(w.rec) - 1}, "
                f"color by {w.display.color_by}, {len(w.pipeline)} steps, {len(w.layers)} layers>")

    # navigation

    @property
    def index(self) -> int:
        return self._w.index

    def goto(self, i: int) -> Frame | None:
        """Show frame ``i`` (loads it now if needed) and return it as loaded."""
        self._w.show_frame(int(i), wait=True)
        return self._w.raw

    def next(self, n: int = 1) -> Frame | None:
        return self.goto(self._w.index + n)

    def prev(self, n: int = 1) -> Frame | None:
        return self.goto(self._w.index - n)

    def play(self, speed: float | None = None) -> None:
        if speed is not None:
            self.speed = speed
        self._w.player.play()

    def pause(self) -> None:
        self._w.player.pause()

    @property
    def playing(self) -> bool:
        return self._w.player.playing

    @property
    def speed(self) -> float:
        """Playback speed: 1 = recorded rate, 0 = as fast as frames load."""
        return self._w.player.speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._w.player.set_speed(float(value))
        self._w.player_bar.set_speed(float(value))

    # data

    @property
    def rec(self):
        return self._w.rec

    @property
    def frame(self) -> Frame | None:
        """Current frame as loaded."""
        return self._w.raw

    @property
    def output(self) -> Frame | None:
        """Current frame after the processing steps."""
        return self._w.output

    @property
    def shown(self) -> Frame | None:
        """What is drawn: the output without hidden echoes."""
        return self._w.shown

    # display

    def color_by(self, name: str, cmap: str | None = None, clim=None) -> None:
        """Color by a field, "range", "height" or "solid color"."""
        d = self._w.display
        if cmap is not None:
            d.cmaps[name] = cmap
        if clim is not None:
            d.clims[name] = (float(clim[0]), float(clim[1]))
        self._w.set_color_by(name)

    @property
    def point_size(self) -> float:
        return self._w.display.point_size

    @point_size.setter
    def point_size(self, size: float) -> None:
        self._w.display.point_size = float(size)
        self._w.apply_display()

    def camera(self, preset: str = "3d") -> None:
        """"3d", "bev", "front", "side" or "fit"."""
        self._w.set_camera(preset)

    def _present_echoes(self) -> set[int]:
        raw = self._w.raw
        if raw is None or "echo" not in raw.fields:
            return set()
        return set(np.unique(raw.echo).tolist())

    @property
    def echoes(self) -> set[int]:
        """Echoes that are drawn; assign a set to hide the others."""
        return self._present_echoes() - self._w.display.hidden_echoes

    @echoes.setter
    def echoes(self, visible) -> None:
        self._w.display.hidden_echoes = self._present_echoes() - {int(e) for e in visible}
        self._w.apply_display()

    def show_removed(self, on: bool = True) -> None:
        """Draw the points the processing steps removed, in red."""
        self._w.display.show_removed = on
        self._w.apply_display()

    def split(self, on: bool = True) -> None:
        """Raw points in the left half, the pipeline's output in the right; one camera."""
        self._w.toggle_split(on)

    # analysis

    def tool(self, name: str = "navigate") -> None:
        """Mouse tool in the 3D view: "navigate", "pick", "measure" or "select"."""
        self._w.set_tool(name)

    @property
    def picked(self) -> dict | None:
        """Values of the point picked last (tool "pick")."""
        return self._w.picked

    @property
    def measurement(self) -> dict | None:
        """Distance between the last two points clicked with the "measure" tool."""
        return self._w.measurement

    @property
    def selection(self) -> Frame | None:
        """Points of ``view.shown`` inside the box selection (tool "select"); the box keeps
        its place relative to the sensor, so this follows the frames."""
        return self._w.selected

    def clear_selection(self) -> None:
        """Clear the picked point, the measurement and the box selection."""
        self._w.clear_tools()

    def bookmark(self, note: str = "", frame: int | None = None) -> None:
        """Bookmark the current frame (or ``frame``) with ``note``."""
        self._w.add_bookmark(note, frame)

    @property
    def bookmarks(self) -> list[tuple[int, str]]:
        b = self._w.bookmarks
        return [(f, b.note(f)) for f in b.frames()] if b is not None else []

    @property
    def metrics(self) -> dict[str, np.ndarray]:
        """Per-frame metrics of the test run (NaN until measured): play, or compute_metrics()."""
        return dict(self._w.metrics.values) if self._w.metrics is not None else {}

    def compute_metrics(self) -> None:
        """Measure every frame in the background; the metric strip fills in."""
        self._w.compute_metrics()

    def save_workspace(self, path) -> Path:
        from . import workspace

        return workspace.save(self._w, path)

    def open_workspace(self, path) -> None:
        from . import workspace

        workspace.load(self._w, path)

    def export(self, out_dir=None, frames=None, fmt: str = "npz", workers: int | None = None,
               compress: bool = True, overwrite: bool = False):
        """Run the current steps over the test run (or ``frames``, e.g. ``range(100, 200)``)
        and write the frames to ``out_dir`` in the background, like File > Export Test Run.
        Returns the job: ``job.done``, ``job.report.summary()``, ``job.cancel()``. Without
        ``out_dir`` a new folder under data/exports is used."""
        from .export_dialog import default_output

        out = Path(out_dir) if out_dir else default_output(self._w.rec.name)
        return self._w.start_export(out, frames, fmt, workers, compress, overwrite)

    # layers

    def highlight(self, func: Callable | None = None, *, name: str | None = None,
                  color: str = "red", size: float = 4.0):
        """Paint points on every frame without removing them; usable as a decorator.

        ``func(frame)`` gets the drawn frame and returns a boolean mask, a Frame or (N, 3) points.
        """

        def register(f: Callable) -> Callable:
            layer_name = name or getattr(f, "__name__", "highlight")
            self._w.add_layer(Layer(layer_name, color, size, func=f))
            return f

        return register if func is None else register(func)

    def show(self, points, name: str = "overlay", color: str = "yellow", size: float = 5.0) -> None:
        """Draw fixed points: a Frame, an (N, 3) array, or a mask over ``view.shown``."""
        if isinstance(points, Frame):
            xyz = points.xyz
        else:
            arr = np.asarray(points)
            if arr.dtype == bool:
                shown = self._w.shown
                if shown is None or arr.shape != (len(shown),):
                    raise ValueError("a mask must have one value per point of view.shown")
                xyz = shown.xyz[arr]
            elif arr.ndim == 2 and arr.shape[1] == 3:
                xyz = arr
            else:
                raise ValueError("expected a Frame, (N, 3) points or a mask, "
                                 f"got shape {arr.shape}")
        self._w.add_layer(Layer(name, color, size, points=np.ascontiguousarray(xyz, np.float32)))

    def remove(self, name: str) -> None:
        self._w.remove_layer(name)

    def clear_layers(self) -> None:
        self._w.clear_layers()

    @property
    def layers(self) -> list[str]:
        return list(self._w.layers)

    # misc

    def refresh(self) -> None:
        """Run the steps on the current frame again and redraw."""
        self._w.refresh()

    def screenshot(self, path=None):
        """Save the 3D view as PNG (returns the path) or, without a path, return it as an array."""
        if path is None:
            return self._w.view.plotter.screenshot(return_img=True)
        self._w.view.save_screenshot(Path(path))
        return Path(path)

    def open(self, path) -> None:
        """Open another test run folder."""
        self._w.open_run(str(path))

    @property
    def color_modes(self) -> list[str]:
        """What ``color_by`` accepts for the open test run."""
        combo = self._w.properties.color_by
        return [combo.itemText(i) for i in range(combo.count())]
