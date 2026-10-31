"""Command line: ``lidarlab [RUN]`` opens the app, ``lidarlab run ...`` processes a test run.

    lidarlab data/synthetic/city_rain
    lidarlab run data/synthetic/city_rain data/exports/clean -p rain_denoise.yaml
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["run"]:
        from .core.batch import main as run

        return run(argv[1:])
    from .app.main import main as app

    return app(argv)
