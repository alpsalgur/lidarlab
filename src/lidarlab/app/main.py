"""Start the application: ``lidarlab [RUN_FOLDER]`` or ``python -m lidarlab``."""

from __future__ import annotations

import argparse
import os
import sys


def make_app(name: str = "LidarLab"):
    """QApplication with the dark theme; ``name`` selects where settings are stored."""
    os.environ.setdefault("QT_API", "pyside6")
    os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")  # ipykernel's debugpy warning
    from PySide6 import QtCore, QtWidgets

    from . import theme

    QtCore.QCoreApplication.setOrganizationName("LidarLab")
    QtCore.QCoreApplication.setApplicationName(name)
    # the UI is English: use '.' decimals in spin boxes, like everywhere else in the app
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.Language.English,
                                             QtCore.QLocale.Country.UnitedStates))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    theme.apply_theme(app)
    app.setWindowIcon(theme.app_icon())
    return app


def _log_without_console() -> None:
    """lidarlab.exe has no console: stdout and stderr are None, so write them to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from .paths import log_dir

    log_dir().mkdir(parents=True, exist_ok=True)
    stream = open(log_dir() / "lidarlab.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def _vtk_messages_to_file() -> None:
    """VTK would open its own message window on Windows; write the messages to a file."""
    from vtkmodules.vtkCommonCore import vtkFileOutputWindow, vtkOutputWindow

    from .paths import log_dir

    log_dir().mkdir(parents=True, exist_ok=True)
    out = vtkFileOutputWindow()
    out.SetFileName(str(log_dir() / "vtk.log"))
    out.AppendOn()
    vtkOutputWindow.SetInstance(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lidarlab", description="LiDAR test run viewer with an integrated Python IDE.")
    parser.add_argument("run", nargs="?", help="test run folder (.npz frames) to open")
    parser.add_argument("--reset-layout", action="store_true",
                        help="start with the default window layout")
    args = parser.parse_args(argv)
    _log_without_console()
    app = make_app()
    from .main_window import MainWindow

    _vtk_messages_to_file()
    window = MainWindow(run=args.run, restore_layout=not args.reset_layout)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
