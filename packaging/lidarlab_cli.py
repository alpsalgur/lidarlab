"""Entry point of lidarlab-cli.exe: ``lidarlab-cli run SOURCE OUT -p preset.yaml`` (console)."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()  # worker processes start this executable again
    from lidarlab.cli import main

    raise SystemExit(main())
