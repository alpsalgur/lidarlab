"""Where the application looks for test runs and scripts, and where it writes logs.

From a source checkout everything lives in the repository; the installed application
(``LidarLab.exe``) keeps test runs, scripts and exports in Documents\\LidarLab instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def frozen() -> bool:
    """True in the packaged application."""
    return bool(getattr(sys, "frozen", False))


def user_dir() -> Path:
    """Documents\\LidarLab (Documents may be redirected, e.g. to OneDrive)."""
    from PySide6 import QtCore

    documents = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(documents or Path.home() / "Documents") / "LidarLab"


def project_dir() -> Path:
    """The repository root for an editable install, Documents\\LidarLab when installed, else
    the working directory."""
    if frozen():
        return user_dir()
    repo = Path(__file__).resolve().parents[3]
    return repo if (repo / "pyproject.toml").is_file() else Path.cwd()


def data_dir() -> Path:
    return project_dir() / "data"


def scripts_dir() -> Path:
    return project_dir() / "scripts"


def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


def log_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "LidarLab"
