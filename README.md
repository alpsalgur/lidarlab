# LidarLab

A desktop viewer for LiDAR test runs stored as one `.npz` file per frame, with an
integrated Python shell and script editor for working on the point clouds: echo filtering,
noise reduction, your own processing steps.

- Play a test run like a video: background loading, a seek bar that shows what is in memory,
  0.25x–4x and max speed, loop
- Color by any field (intensity, echo, range, height, ring, ground-truth labels), camera presets
  (3D, bird's eye, driver, side), range rings and an ego vehicle outline
- 12 built-in filters with sliders, applied to every frame while playing; heavy filters run
  ahead of the playhead in background threads
- Split view: raw points on the left, the pipeline's output on the right, one camera
- An IPython shell and a script editor running in the viewer's process: scripts see the frames
  as NumPy arrays, without copies
- Export: run the pipeline over a whole test run (or a range) from the app, in the background,
  or with `lidarlab run` on the command line; both write the same files (npz in the source's
  layout, or PLY/PCD)
- Pipeline presets in YAML, converters for KITTI, Boreas (snow) and WADS (labeled snow), and a
  synthetic multi-echo recording with rain, spray and per-point ground truth

Built with Python 3.11+, PySide6 (Qt 6), VTK (through PyVista), IPython/qtconsole, NumPy and
SciPy.

## Installation

```powershell
python -m venv venv
venv\Scripts\python -m pip install -e ".[dev]"
venv\Scripts\python tools\smoke_test.py      # checks that Qt, VTK and the shell work together
```

## Running

```powershell
.\LidarLab.bat                                    # or double-click it in Explorer (no console)
venv\Scripts\lidarlab data\synthetic\city_rain   # from a terminal, with a test run
```

The app remembers the last test run and the window layout; `--reset-layout` restores the
default layout.

| Area | Contents |
|---|---|
| Left | **Test Runs**: the recordings under `data/` (double-click opens, `+` adds another folder) · **Pipeline Browser**: the test run, its processing steps and layers; `+ Filter` adds a built-in filter, the folder and disk icons load and save presets; check boxes switch steps on and off, the columns show the points left after each step and its time |
| Center | 3D view and player bar. The light gray part of the seek bar marks frames in memory, the green line under it frames the pipeline has processed |
| Right | **Properties**: the selected step's parameters, coloring, point size, echo visibility, split view, scene, camera · **Information**: file, time, pose, points per echo, bounds, arrays, timings |
| Bottom | **Python Shell** (IPython in the viewer's process) · **Script Editor** |

Keyboard: Space play/pause · ←/→ one frame (Shift: 10) · Home/End · 1–4 cameras (3D, bird's eye,
driver, side) · R fit · C next color field · +/− point size · Ctrl+T split view · F1 scripting
reference. While typing in the shell, the editor or a field, these keys type as usual.

## Filters

Pipeline Browser → `+ Filter` (or the **Filters** menu) adds a built-in filter and selects it;
its parameters appear under Properties as sliders. While a slider moves, the result is computed
in the background and drawn when ready, so the window does not freeze. During playback the
steps run on the frames ahead of the playhead in three threads, and the results are kept in
memory.

| Group | Filters |
|---|---|
| Echo | echo selection (drop the cover echo, first/last/strongest real echo, echo n) · weak first echoes (rain, spray: a weak first echo with a later echo behind it) |
| Basic | threshold (any field, `range`, `z`, …) · crop box (e.g. the own vehicle) |
| Geometry | level with roll/pitch · voxel downsampling |
| Noise | SOR · ROR · DROR · DSOR · LIOR |
| Ground | RANSAC ground plane (remove it, or keep only the ground) |

Split view (Ctrl+T) shows the raw points on the left and the pipeline's output on the right with
one camera; "Show removed points" draws what the pipeline removed in red. A ready-made preset:
Filters → Load Pipeline Preset → `examples/presets/rain_denoise.yaml` (cover echo + weak first
echoes + DROR). On the labeled synthetic run it removes 94–96 % of the rain and spray, and
93–94 % of the points it removes are noise.

## Export

File → Export Test Run (Ctrl+Shift+E) runs a copy of the current steps over all frames or a
range and writes them to a new folder, in the background with a progress window; sliders moved
meanwhile do not change the export. The same from a terminal, with a preset:

```powershell
venv\Scripts\lidarlab run data\synthetic\city_rain data\exports\clean -p rain_denoise.yaml
venv\Scripts\lidarlab run data\kitti\2011_09_26_drive_0011 data\exports\kitti_ply --format ply
venv\Scripts\lidarlab run data\boreas\boreas-2021-01-26-11-22_3300 data\exports\snow_pcd --frames 50:150 --format pcd
```

Options: `--frames A:B` (A included, B not), `--format npz|ply|pcd`, `--workers N`,
`--processes`, `--no-compress`, `--overwrite`. The exit code is 0 when every frame was written.

The output folder is a test run again: one file per source frame under the same name, npz in
the source's layout (in echo-axis files removed points become `NaN`), plus `recording.json`
(what was done), `pipeline.yaml` (the steps) and `export.log` (points in and out, time and error
per frame). A frame whose steps fail is not written, so an unfiltered frame cannot slip into the
output. PLY and PCD are binary and carry every numeric field (intensity, echo, ring, label…).

On 4 cores / 8 threads, the rain preset exports a 600-frame run (~300k points per frame) in
about 90 s (6.8 frames/s), from the app and from the command line alike, with identical files.

## Working with Python

The shell and the editor share their names with the viewer: `rec` (test run), `frame` (current
frame as loaded), `view` (the viewer), `pipe` (processing steps), `lab`, `np`.

```python
frame.fields                       # intensity, echo, ring, label
frame[frame.echo != 0]             # subset: all fields and source indices are kept
frame.range, frame.pulse           # derived: distance, echoes of the same laser pulse

@lab.step                          # applied to every frame while playing
def remove_cover_echo(f):
    return f.echo != 0             # True = keep the point

@lab.filter(radius=(0.05, 2.0), k=(1, 20))   # with parameters: sliders under Properties
def my_ror(f, radius=0.4, k=2):              # defaults come from the signature
    ...

@view.highlight(color="red")       # paints without removing, recomputed on every frame
def weak(f):
    return f.intensity < 10

pipe.add(lab.filters.DROR(beta=3))  # built-in filter;  pipe["dror"].set(min_neighbors=3)
pipe.save("denoise.yaml")           # preset: steps, order, values
view.split()                        # raw | after the pipeline
view.output.save("out.npz")         # the frame after the steps, in the source layout
job = view.export("data/exports/clean", frames=range(100, 200))   # in the background
```

Examples: Script Editor → Examples (exploring a frame, cover echo, rain echoes, bright points,
ROR, export, a filter of your own with sliders, scoring the pipeline against labels). F5 runs the
script, Ctrl+Enter the selection or line. With "Run on external save" on, saving a script that
is open in the editor from another editor (e.g. VS Code) reloads and runs it; steps are replaced
by name, and values tuned in the app are kept unless the default changed.

Without the GUI:

```python
import lidarlab as lab

rec = lab.open("data/synthetic/city_rain")
pipe = lab.Pipeline()
pipe.load("src/lidarlab/examples/presets/rain_denoise.yaml")
clean = pipe(rec[150])
report = lab.export_run(rec, "data/exports/clean", pipe)   # blocks until done
print(report.summary())
```

With `processes=True` on Windows, call `export_run` under `if __name__ == "__main__":`, since
each worker process imports the script again.

## Test data

`data/` is not under version control; these commands rebuild it.

```powershell
# synthetic run: 200 frames, 3 echoes, rain and spray from the middle on; every point has a
# ground-truth label (cover, ground, object, edge, vegetation, rain, spray)
venv\Scripts\python -m lidarlab.datasets.synthetic data\synthetic\city_rain
# the same at full length: 600 frames, 2.9 GB
venv\Scripts\python -m lidarlab.datasets.synthetic data\synthetic\city_rain_600 --frames 600

# KITTI raw drive: downloads ~0.9 GB and converts it
venv\Scripts\python -m lidarlab.datasets.kitti 2011_09_26_drive_0011 data\kitti

# Boreas: 20 s in heavy snowfall (128-beam Velodyne Alpha Prime), downloads ~1 GB
venv\Scripts\python -m lidarlab.datasets.boreas boreas-2021-01-26-11-22 data\boreas --start 3300

# WADS: real data with point-wise snow labels. Its download page needs a browser:
# get a sequence from https://digitalcommons.mtu.edu/wads/, unzip it, then
venv\Scripts\python -m lidarlab.datasets.wads C:\Downloads\wads_seq_11 data\wads\seq_11
```

Licenses: KITTI CC BY-NC-SA 3.0 (non-commercial), Boreas and WADS CC BY 4.0.

### npz layouts

The reader recognizes:

- **Echo axis:** point keys of shape `(E, P)`, row = echo, column = pulse, `NaN` = no return.
  Keys of shape `(P,)` belong to the pulse (e.g. `ring`).
- **Flat:** valid points only, shape `(M,)`, the echo number in `echo_number`.
- **Points:** one `points` array of shape `(M, 3+)`.

Scalars such as `roll`, `pitch`, `yaw` and `timestamp` become frame metadata. Writing a frame
restores the layout it was read from; in echo-axis files removed points become `NaN`.

## Development

```powershell
venv\Scripts\python -m pytest
venv\Scripts\python -m ruff check .
# drives the window without a user (a window opens for about 1.5 minutes) and writes
# screenshots to data\previews
venv\Scripts\python tools\ui_selftest.py data\synthetic\city_rain data\kitti\2011_09_26_drive_0011
```
