"""``LidarLab --self-check REPORT.json``: open the window on a small generated test run, use
the main features once and write what worked. Tests the packaged application without a user.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path


def run_self_check(report_path) -> int:
    from PySide6 import QtCore

    import lidarlab

    from .main import make_app
    from .sample_run import SampleJob

    report_path = Path(report_path)
    checks: dict[str, bool] = {}
    info: dict[str, str] = {"lidarlab": lidarlab.__version__, "python": sys.version.split()[0],
                            "frozen": str(bool(getattr(sys, "frozen", False)))}
    scratch = Path(tempfile.mkdtemp(prefix="lidarlab-selfcheck-"))
    # settings in an INI file in the scratch folder: the check leaves nothing in the registry
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope,
                             str(scratch / "settings"))
    app = make_app("LidarLab-selfcheck")
    window = None

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            time.sleep(0.005)

    def wait(condition, timeout: float = 60.0) -> bool:
        end = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > end:
                return False
            pump(0.05)
        return True

    try:
        # the test run comes from File > Create Sample Test Run's job: a thread that writes the
        # frames in worker processes (in the installed application, LidarLab.exe started again)
        run = scratch / "run"
        job = SampleJob(run, frames=4)
        generated = []
        job.finished.connect(generated.append)
        job.start()
        wait(lambda: bool(generated), 120)
        if generated and isinstance(generated[0], Exception):
            info["sample run error"] = repr(generated[0])
        checks["sample test run generated"] = (generated == [run]
                                               and len(list(run.glob("*.npz"))) == 4)
        from .main_window import MainWindow

        window = MainWindow(run=str(run), restore_layout=False)
        window.resize(1280, 800)
        window.show()
        checks["test run opens and a frame is shown"] = wait(lambda: window.raw is not None)
        window.shell.execute("selfcheck_points = len(frame)")
        checks["Python shell runs code on the frame"] = wait(
            lambda: window.shell.namespace.get("selfcheck_points") == len(window.raw), 20)
        import jedi

        names = [c.name for c in jedi.Interpreter("frame.", [window.shell.namespace])
                 .complete(1, 6)]
        checks["editor completion sees the frame's fields"] = "intensity" in names
        window.add_filter("dror")
        step = window.pipeline["dror"]
        wait(lambda: step.n_out is not None or step.error is not None, 30)
        checks["a filter runs on the frame"] = step.error is None and step.n_out is not None
        widget = window.view.plotter.interactor
        render_window = window.view.plotter.render_window
        # on the first start after installing, Windows scans the new DLLs and the window can
        # take a while to appear
        start = time.monotonic()
        wait(lambda: min(render_window.GetSize()) > 0, 90)
        info["3D view on screen after"] = f"{time.monotonic() - start:.1f} s"
        info["render window"] = (f"{render_window.GetClassName()} size {render_window.GetSize()}"
                                 f" mapped={render_window.GetMapped()} widget visible="
                                 f"{widget.isVisible()} {widget.width()}x{widget.height()}")
        try:
            image = window.view.capture()
            checks["the 3D view renders"] = float(image.std()) > 1.0
        except RuntimeError as e:
            info["render error"] = str(e)
            checks["the 3D view renders"] = False
        report = lidarlab.export_run(run, scratch / "export", window.pipeline.snapshot(),
                                     workers=2)
        checks["export writes every frame"] = report.written == 4 and not report.failed
        import numpy
        from PySide6 import __version__ as pyside
        from vtkmodules.vtkCommonCore import vtkVersion

        info.update(pyside6=pyside, vtk=vtkVersion.GetVTKVersion(), numpy=numpy.__version__)
        renderer = window.view.plotter.render_window.ReportCapabilities()
        info["opengl"] = next((line.split(":", 1)[1].strip() for line in renderer.splitlines()
                               if "renderer string" in line.lower()), "unknown")
    except Exception:  # noqa: BLE001 - the report says what failed
        info["error"] = traceback.format_exc()
    finally:
        if window is not None:
            window.close()
            pump(0.2)
            window.settings.sync()  # now, not at exit into the deleted scratch folder
        shutil.rmtree(scratch, ignore_errors=True)
    ok = bool(checks) and all(checks.values()) and "error" not in info
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"ok": ok, "checks": checks, "info": info}, indent=2),
                           encoding="utf-8")
    return 0 if ok else 1
