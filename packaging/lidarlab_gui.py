"""Entry point of LidarLab.exe (the packaged application, no console window)."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()  # worker processes start this executable again
    from lidarlab.app.main import main

    raise SystemExit(main())
