"""Where the application looks for test runs and scripts, and where it writes logs."""

from __future__ import annotations

import os
from pathlib import Path


def project_dir() -> Path:
    """The repository root for an editable install, else the working directory."""
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
