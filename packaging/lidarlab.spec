# PyInstaller build of LidarLab: one folder with LidarLab.exe (the application, no console)
# and lidarlab-cli.exe (``lidarlab-cli run ...``) sharing the libraries.
# Run through tools/build_release.py, which also makes the icon and the version resource.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 - defined by PyInstaller
BUILD = ROOT / "build"

hidden = collect_submodules("lidarlab") + [
    "ipykernel.inprocess.ipkernel",  # the shell's in-process kernel, imported by name
    "qtconsole.inprocess",
]
datas = collect_data_files("lidarlab", includes=["examples/*.py", "examples/presets/*.yaml"])
datas += collect_data_files("qtconsole")
excludes = [
    "tkinter", "_tkinter",  # no Tk user interface
    "debugpy",  # ipykernel's debugger; the in-process shell runs without it
    "pytest", "ruff", "PyQt5", "PyQt6", "PySide2",
    "matplotlib.backends.backend_tkagg", "matplotlib.backends.backend_webagg",
]


# jedi's stubs for third-party packages (django, oauthlib, …): not needed for completing frames
# and NumPy, ~4,800 files and the deepest paths of the build (installs into long folder names
# would pass Windows' 260-character limit). The standard library stubs stay.
SKIP_DATA = ("jedi/third_party/typeshed/stubs/", "jedi/third_party/django-stubs/")


def analysis(script: str):
    a = Analysis(  # noqa: F821
        [str(ROOT / "packaging" / script)],
        pathex=[str(ROOT / "src")],
        hiddenimports=hidden,
        datas=datas,
        excludes=excludes,
    )
    a.datas = [d for d in a.datas if not any(s in d[0].replace("\\", "/") for s in SKIP_DATA)]
    return a


def executable(a, name: str, console: bool):
    return EXE(  # noqa: F821
        PYZ(a.pure),  # noqa: F821
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        console=console,
        icon=str(BUILD / "lidarlab.ico"),
        version=str(BUILD / "version_info.txt"),
        upx=False,
    )


gui = analysis("lidarlab_gui.py")
cli = analysis("lidarlab_cli.py")
coll = COLLECT(  # noqa: F821
    executable(gui, "LidarLab", console=False), gui.binaries, gui.datas,
    executable(cli, "lidarlab-cli", console=True), cli.binaries, cli.datas,
    name="LidarLab",
    upx=False,
)
