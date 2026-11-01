"""Build the Windows release and check that the built application works.

1. icon and version resource (build/)
2. PyInstaller: dist/LidarLab/ with LidarLab.exe (no console) and lidarlab-cli.exe
3. checks: LidarLab.exe --self-check, lidarlab-cli.exe run with threads and with processes
4. Inno Setup: dist/LidarLab-<version>-Setup.exe (skipped when ISCC.exe is not installed)

    python tools/build_release.py [--no-installer] [--skip-build]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD, DIST = ROOT / "build", ROOT / "dist"
APP = DIST / "LidarLab"
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def app_env() -> dict:
    """Environment for the built executables: without the offscreen Qt platform make_icon
    uses, or the 3D view gets no window and VTK crashes."""
    return {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}


def make_icon(path: Path) -> None:
    """The app icon (an SVG in theme.py) as a multi-size .ico."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window for this process
    from PIL import Image
    from PySide6 import QtCore, QtWidgets

    from lidarlab.app import theme

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pixmap = theme.app_icon().pixmap(256, 256)
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    image = Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA")
    image.save(path, sizes=[(s, s) for s in ICON_SIZES])


def write_version_info(path: Path, version: str) -> None:
    numbers = tuple(int(p) for p in version.split(".")[:3]) + (0,)
    strings = {"FileDescription": "LidarLab", "ProductName": "LidarLab",
               "FileVersion": version, "ProductVersion": version, "InternalName": "LidarLab"}
    table = ", ".join(f"StringStruct({k!r}, {v!r})" for k, v in strings.items())
    path.write_text(
        "VSVersionInfo(\n"
        f"  ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}, mask=0x3f, flags=0x0,\n"
        "                    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),\n"
        f"  kids=[StringFileInfo([StringTable('040904B0', [{table}])]),\n"
        "        VarFileInfo([VarStruct('Translation', [1033, 1200])])]\n"
        ")\n", encoding="utf-8")


def pyinstaller() -> None:
    command = [sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "lidarlab.spec"),
               "--noconfirm", "--clean", "--log-level", "WARN",
               "--distpath", str(DIST), "--workpath", str(BUILD / "pyinstaller")]
    subprocess.run(command, check=True, cwd=ROOT)


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def check_app() -> bool:
    report = BUILD / "selfcheck.json"
    report.unlink(missing_ok=True)
    start = time.perf_counter()
    subprocess.run([str(APP / "LidarLab.exe"), "--self-check", str(report)], timeout=300,
                   env=app_env())
    if not report.exists():
        log("LidarLab.exe --self-check wrote no report; see the log in "
            "%LOCALAPPDATA%\\LidarLab\\lidarlab.log")
        return False
    data = json.loads(report.read_text(encoding="utf-8"))
    for name, ok in data["checks"].items():
        log(f"  {'OK  ' if ok else 'FAIL'} {name}")
    if "error" in data["info"]:
        log(data["info"]["error"])
    log(f"LidarLab.exe self-check: {'passed' if data['ok'] else 'FAILED'} in "
        f"{time.perf_counter() - start:.0f} s ({data['info'].get('opengl', '?')})")
    return data["ok"]


def check_cli() -> bool:
    from lidarlab.datasets import synthetic

    run = BUILD / "check_run"
    shutil.rmtree(run, ignore_errors=True)
    synthetic.generate(run, synthetic.Config(frames=6, n_beams=32, n_azimuth=512), log=None)
    ok = True
    for extra in ([], ["--processes", "--workers", "2"]):
        out = BUILD / "check_out"
        shutil.rmtree(out, ignore_errors=True)
        # the built-in preset by name: found inside the packaged application
        result = subprocess.run([str(APP / "lidarlab-cli.exe"), "run", str(run), str(out),
                                 "-p", "rain_denoise", *extra], capture_output=True, text=True,
                                timeout=300, env=app_env())
        written = len(list(out.glob("*.npz")))
        passed = result.returncode == 0 and written == 6
        mode = "processes" if extra else "threads"
        log(f"lidarlab-cli.exe run ({mode}): exit {result.returncode}, {written}/6 frames "
            f"{'OK' if passed else 'FAILED'}")
        if not passed:
            log(result.stdout[-2000:] + result.stderr[-2000:])
        ok &= passed
    return ok


def find_iscc() -> Path | None:
    candidates = [shutil.which("ISCC.exe")]
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            candidates.append(os.path.join(base, "Inno Setup 6", "ISCC.exe"))
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


def installer(version: str) -> Path | None:
    iscc = find_iscc()
    if iscc is None:
        log("Inno Setup (ISCC.exe) not found: no installer. winget install JRSoftware.InnoSetup")
        return None
    subprocess.run([str(iscc), "/Q", f"/DAppVersion={version}",
                    str(ROOT / "packaging" / "installer.iss")], check=True, cwd=ROOT)
    return DIST / f"LidarLab-{version}-Setup.exe"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-installer", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="check an existing dist/LidarLab")
    a = ap.parse_args()
    from lidarlab import __version__ as version

    BUILD.mkdir(exist_ok=True)
    if not a.skip_build:
        start = time.perf_counter()
        make_icon(BUILD / "lidarlab.ico")
        write_version_info(BUILD / "version_info.txt", version)
        log(f"PyInstaller, LidarLab {version} …")
        pyinstaller()
        log(f"dist/LidarLab: {folder_size(APP) / 2**20:,.0f} MB in "
            f"{time.perf_counter() - start:.0f} s")
    ok = check_app() & check_cli()
    if not ok:
        log("checks failed: no installer")
        return 1
    if not a.no_installer:
        setup = installer(version)
        if setup is not None:
            log(f"installer: {setup} ({setup.stat().st_size / 2**20:,.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
