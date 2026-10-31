"""Drive the main window without a user, check what it does and save screenshots.

Opens a test run, plays it at 1x and as fast as frames load, uses the keyboard, runs the
example scripts through the script editor and the shell, then opens a second test run.
A window appears for about a minute.

    python tools/ui_selftest.py SYNTHETIC_RUN [OTHER_RUN] [--out data/previews]
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtTest

from lidarlab.app.main import make_app

Key = QtCore.Qt.Key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("other", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, default=Path("data/previews"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    app = make_app("LidarLab-selftest")  # settings apart from the real application
    QtCore.QSettings().clear()
    from lidarlab.app.main_window import MainWindow
    from lidarlab.app.paths import examples_dir

    w = MainWindow(run=str(args.run), restore_layout=False)
    w.resize(1720, 1060)
    w.show()
    checks: dict[str, bool] = {}
    numbers: dict[str, str] = {}
    displayed = [0]
    display = w._display

    def counting_display(raw):
        displayed[0] += 1
        display(raw)

    w._display = counting_display

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 10)
            time.sleep(0.001)

    def wait(condition, timeout: float = 30.0) -> bool:
        end = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > end:
                return False
            pump(0.02)
        return True

    def shot(name: str) -> None:
        pump(0.4)
        w.grab_window().save(str(args.out / name))

    def play_for(seconds: float, speed: float) -> tuple[float, int]:
        w.api.speed = speed
        w.player.stalls = 0
        displayed[0] = 0
        w.player.play()
        start = time.perf_counter()
        pump(seconds)
        w.player.pause()
        return displayed[0] / (time.perf_counter() - start), w.player.stalls

    def run_example(name: str) -> None:
        w.editor.open_example(examples_dir() / name)
        w.editor.run_current()
        pump(0.4)  # let the console show the output before the next command

    # open and play
    checks["first frame shown"] = wait(lambda: w.raw is not None)
    pump(1.0)
    shot("ui_01_opened.png")
    fps, stalls = play_for(6.0, 1.0)
    numbers["playback 1x"] = (f"{fps:.1f} fps (recorded {w.rec.rate_hz:.1f} Hz), "
                              f"{stalls} waits for loading")
    checks["1x keeps the recorded rate"] = fps > 0.9 * w.rec.rate_hz
    fps, _ = play_for(6.0, 0.0)
    numbers["playback max"] = f"{fps:.1f} fps"
    load = np.median(list(w.rec.load_ms.values()))
    numbers["read one frame"] = f"{load:.0f} ms median ({len(w.rec.load_ms)} frames read)"
    w.api.speed = 1.0

    # keyboard
    vtk_widget = w.view.plotter.interactor
    QtTest.QTest.keyClick(vtk_widget, Key.Key_Space)
    started = w.player.playing
    QtTest.QTest.keyClick(vtk_widget, Key.Key_Space)
    checks["Space in the 3D view plays and pauses"] = started and not w.player.playing
    console = w.shell.widget._control
    console.setFocus()
    QtTest.QTest.keyClick(console, Key.Key_Space)
    checks["Space in the shell is just typing"] = not w.player.playing
    QtTest.QTest.keyClick(vtk_widget, Key.Key_Right)
    before = w.index
    QtTest.QTest.keyClick(vtk_widget, Key.Key_Right)
    checks["→ steps one frame"] = w.index == before + 1

    # scripts on a rainy frame
    w.api.goto(150)
    raw = w.raw
    w.set_color_by("echo")
    run_example("01_explore_frame.py")
    shot("ui_02_echoes.png")

    run_example("02_remove_cover_echo.py")
    checks["cover echo step registered"] = wait(lambda: "remove_cover_echo" in w.pipeline)
    checks["cover echo removed"] = (w.output is not None and len(w.output) < len(raw)
                                    and not np.any(w.output.echo == 0))
    run_example("03_remove_rain_echoes.py")
    checks["rain step registered"] = wait(lambda: "remove_rain_echoes" in w.pipeline)
    rain = w.pipeline["remove_rain_echoes"]
    checks["rain echoes removed"] = rain.error is None and rain.n_out < rain.n_in
    run_example("04_highlight_brightest.py")
    checks["highlight layer drawn"] = (wait(lambda: "brightest" in w.layers)
                                       and w.layers["brightest"].count > 0)
    numbers["steps on frame 150"] = "; ".join(
        f"{s.name}: {s.n_in:,} -> {s.n_out:,} in {s.ms:.1f} ms" for s in w.pipeline)
    import jedi

    names = [c.name for c in jedi.Interpreter("frame.", [w.shell.namespace]).complete(1, 6)]
    checks["completion lists the frame's fields"] = {"intensity", "echo", "ring"} <= set(names)
    w.display.show_removed = True
    w.set_color_by("intensity")
    w.shell.execute("pipe")
    shot("ui_03_rain_removed.png")
    w.set_camera("bev")
    shot("ui_04_bev.png")

    # filters: preset, sliders, split view, background processing
    w.clear_layers()
    w.set_camera("3d")
    preset = examples_dir() / "presets" / "rain_denoise.yaml"
    checks["preset loads"] = w.load_preset(str(preset)) and \
        [s.name for s in w.pipeline] == ["echo", "weak_first_echo", "dror"]
    editor = w.properties.step_editor
    w.pipeline_panel.select_step("dror")
    pump(0.2)
    checks["selected step shows sliders"] = (editor.isVisible() and editor.step_name == "dror"
                                            and editor.widgets["beta"].slider is not None)
    dror = w.pipeline["dror"]
    if "label" in w.raw.fields:  # score the preset against the ground truth
        kept = np.isin(w.raw.index, w.output.index)
        noise = np.isin(w.raw.label, [5, 6])  # rain, spray (the cover echo goes anyway)
        caught = np.count_nonzero(~kept & noise)
        removed = np.count_nonzero(~kept & (w.raw.label != 0))
        numbers["rain preset vs labels"] = (f"recall {caught / noise.sum():.0%}, precision "
                                            f"{caught / max(removed, 1):.0%} on frame {w.index}")
        checks["rain preset removes most rain and spray"] = caught / noise.sum() > 0.9
    run_example("08_score_against_labels.py")
    checks["scoring example runs"] = "missed_noise" in w.layers
    w.remove_layer("missed_noise")
    start_beta = dror.values["beta"]
    removed_before = dror.n_in - dror.n_out
    beta = editor.widgets["beta"]
    start = time.perf_counter()
    for value in range(beta.slider.value(), beta.slider.value() - 150, -10):  # drag to beta ~1.6
        beta.slider.setValue(value)
        pump(0.02)
    drag = time.perf_counter() - start
    checks["slider drag applies"] = wait(
        lambda: abs(dror.values["beta"] - beta.editor.value()) < 1e-6
        and w.preview_pending is None)
    removed_after = dror.n_in - dror.n_out
    numbers[f"DROR beta {start_beta:.1f} -> {dror.values['beta']:.1f}"] = (
        f"removed {removed_before:,} -> {removed_after:,} points (15 slider steps in "
        f"{drag:.1f} s, result shown after {time.perf_counter() - start:.1f} s)")
    checks["smaller radius removes more"] = removed_after > removed_before
    editor._reset()
    checks["reset to defaults"] = (dror.values["beta"] == 3.0
                                   and editor.widgets["beta"].editor.value() == 3.0)
    w.display.show_removed = True
    w.toggle_split(True)
    checks["split view"] = w.view.split and w.view.raw_cloud.count == len(w.raw) - \
        int(np.count_nonzero(np.isin(w.raw.echo, list(w.display.hidden_echoes))))
    camera = w.view.renderer.GetActiveCamera()
    before = camera.GetPosition()
    vtk_widget.setFocus()
    left = QtCore.QPoint(vtk_widget.width() // 4, vtk_widget.height() // 2)
    QtTest.QTest.mousePress(vtk_widget, QtCore.Qt.MouseButton.LeftButton, pos=left)
    for dx in range(10, 90, 10):
        QtTest.QTest.mouseMove(vtk_widget, left + QtCore.QPoint(dx, dx // 4))
        pump(0.01)
    QtTest.QTest.mouseRelease(vtk_widget, QtCore.Qt.MouseButton.LeftButton,
                              pos=left + QtCore.QPoint(80, 20))
    checks["dragging the left half turns the shared camera"] = camera.GetPosition() != before
    w.set_camera("3d")
    shot("ui_06_split_rain_filtered.png")
    w.display.show_removed = False
    w.toggle_split(False)

    w.player.stalls = 0
    fps, stalls = play_for(8.0, 1.0)
    processed = len(w.processor.cached())
    numbers["playback 1x with 3 filters"] = (f"{fps:.1f} fps, {stalls} waits; {processed} frames "
                                              f"processed in the background")
    checks["filters keep playback above 5 fps"] = fps > 5.0
    step_ms = "; ".join(f"{s.name} {s.ms:.0f} ms" for s in w.pipeline if s.ms is not None)
    numbers["step times (shown frame)"] = step_ms

    name = w.add_filter("ground")
    checks["filter added from the menu"] = (name == "ground" and editor.step_name == "ground"
                                            and w.pipeline["ground"].n_out is not None)
    saved = args.out / "selftest_preset.yaml"
    checks["preset saves"] = w.save_preset(str(saved)) and "filter: ground" in saved.read_text()
    w.pipeline.remove("ground")
    run_example("07_filter_with_sliders.py")
    checks["example filter registered"] = wait(lambda: "ring_jumps" in w.pipeline)
    jumps = w.pipeline["ring_jumps"]
    checks["example filter has sliders"] = set(jumps.params) == {"max_jump", "max_range", "echo"}
    numbers["ring_jumps on this frame"] = f"{jumps.n_in:,} -> {jumps.n_out:,} in {jumps.ms:.0f} ms"

    # export: the app (in the background) and the command line write the same frames
    gui_out, cli_out = args.out / "selftest_export_app", args.out / "selftest_export_cli"
    for folder in (gui_out, cli_out):
        shutil.rmtree(folder, ignore_errors=True)
    job = w.start_export(gui_out, frames=range(140, 160), interactive=False)
    checks["export runs in the background"] = not job.done
    checks["export finishes"] = wait(lambda: job.done, timeout=300) and job.error is None
    numbers["export of 20 frames, app"] = job.report.summary() if job.report else str(job.error)
    preset_file = args.out / "selftest_export_preset.yaml"
    w.pipeline.save(preset_file)  # includes ring_jumps, a step defined by a script
    start = time.perf_counter()
    cli = subprocess.run([sys.executable, "-m", "lidarlab", "run", str(args.run), str(cli_out),
                          "-p", str(preset_file), "--frames", "140:160"],
                         capture_output=True, text=True)
    numbers["export of 20 frames, command line"] = (
        f"exit code {cli.returncode} in {time.perf_counter() - start:.1f} s (with start-up)")
    same = cli.returncode == 0 and len(list(gui_out.glob("*.npz"))) == 20
    for path in gui_out.glob("*.npz"):
        with np.load(path) as a, np.load(cli_out / path.name) as b:
            same &= a.files == b.files and all(
                np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
    checks["app and command line write the same frames"] = same

    # a second test run (other layout)
    if args.other:
        w.pipeline.clear()
        w.clear_layers()
        w.display.show_removed = False
        opened = w.open_run(str(args.other))
        checks["second test run opens"] = opened and wait(lambda: w.raw is not None)
        checks["echo boxes follow the run"] = sorted(w.properties.echo_checks) == [1]
        fps, stalls = play_for(4.0, 1.0)
        numbers["other run 1x"] = f"{fps:.1f} fps (recorded {w.rec.rate_hz:.1f} Hz), {stalls} waits"
        w.api.goto(100)
        run_example("04_highlight_brightest.py")
        checks["highlight works on the other run"] = w.layers["brightest"].count > 0
        shot("ui_05_other_run.png")

    w.close()
    for name, value in numbers.items():
        print(f"{name:>30}: {value}")
    for name, ok in checks.items():
        print(f"{'OK' if ok else 'FAIL':>30}: {name}")
    print(f"screenshots in {args.out.resolve()}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
