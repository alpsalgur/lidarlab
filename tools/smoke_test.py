"""Check that Qt, VTK and the in-process IPython console work together.

Opens a window with a 3D point cloud above an IPython console for a few seconds, runs code
in the console's kernel, saves screenshots and prints the library versions.

    python tools/smoke_test.py [OUT_DIR]      (default: data/smoke)
"""

import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtpy import QtCore, QtWidgets

N_POINTS = 300_000
PACKAGES = ("PySide6-Essentials", "vtk", "pyvista", "pyvistaqt", "qtconsole", "ipykernel", "numpy")


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/smoke")
    out.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication(sys.argv)

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("LidarLab smoke test")
    window.resize(1100, 800)
    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
    window.setCentralWidget(splitter)

    plotter = QtInteractor(splitter)
    splitter.addWidget(plotter.interactor)
    rng = np.random.default_rng(0)
    cloud = pv.PolyData((rng.normal(size=(N_POINTS, 3)) * [20, 20, 2]).astype(np.float32))
    cloud["range"] = np.linalg.norm(cloud.points, axis=1)
    plotter.add_mesh(cloud, scalars="range", cmap="viridis", point_size=2)
    plotter.view_isometric()

    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel(show_banner=False)
    kernel_manager.kernel.gui = "qt"
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()
    console = RichJupyterWidget()
    console.kernel_manager = kernel_manager
    console.kernel_client = kernel_client
    splitter.addWidget(console)
    splitter.setSizes([550, 250])
    shell = kernel_manager.kernel.shell
    shell.push({"cloud": cloud, "plotter": plotter, "np": np})

    checks: dict[str, bool] = {}
    info: dict[str, str] = {}

    def run_checks() -> None:
        shell.run_cell("n = cloud.n_points")
        checks["kernel shares the app namespace"] = shell.user_ns.get("n") == N_POINTS
        console.execute("print('console ok:', cloud.n_points, 'points')")
        plotter.screenshot(str(out / "smoke_3d.png"))
        checks["3D view renders"] = (out / "smoke_3d.png").stat().st_size > 0
        caps = plotter.render_window.ReportCapabilities()
        info["OpenGL renderer"] = next(
            (line.split(":", 1)[1].strip() for line in caps.splitlines()
             if "renderer string" in line.lower()),
            "unknown",
        )

    def finish() -> None:
        # widget grab, not a screen grab: the window may open behind other windows
        console.grab().save(str(out / "smoke_console.png"))
        kernel_client.stop_channels()
        kernel_manager.shutdown_kernel()
        plotter.close()
        app.quit()

    window.show()
    window.raise_()
    QtCore.QTimer.singleShot(1500, run_checks)
    QtCore.QTimer.singleShot(3500, finish)
    app.exec()

    print(f"{'Python':>20}: {sys.version.split()[0]}")
    for name in PACKAGES:
        print(f"{name:>20}: {version(name)}")
    for name, value in info.items():
        print(f"{name:>20}: {value}")
    for name, ok in checks.items():
        print(f"{'OK' if ok else 'FAIL':>20}: {name}")
    print(f"screenshots in {out.resolve()}")
    return 0 if len(checks) == 2 and all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
