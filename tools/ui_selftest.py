"""Drive the main window without a user, check what it does and save screenshots.

Opens a test run, plays it at 1x and as fast as frames load, uses the keyboard, runs the
example scripts through the script editor and the shell, then opens a second test run.
A window appears for about a minute.

    python tools/ui_selftest.py SYNTHETIC_RUN [OTHER_RUN] [--out data/previews]
"""

import argparse
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
