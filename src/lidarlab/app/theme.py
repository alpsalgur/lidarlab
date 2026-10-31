"""Dark theme, line icons and small generated icons."""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

TEXT = "#d4d4d4"
MUTED = "#8b949e"
ACCENT = "#3794ff"
WINDOW = "#252526"
BASE = "#1e1e1e"
PANEL = "#2d2d30"
BORDER = "#3c3c3c"
SELECTION = "#094771"
ERROR = "#f48771"

STYLE_SHEET = f"""
QMainWindow::separator {{ background: {PANEL}; width: 4px; height: 4px; }}
QMainWindow::separator:hover {{ background: {ACCENT}; }}
QDockWidget::title {{ background: {PANEL}; padding: 5px 8px; border-bottom: 1px solid {BORDER}; }}
QToolBar {{ background: {PANEL}; border: none; spacing: 3px; padding: 2px 4px; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton {{ border: 1px solid transparent; border-radius: 4px; padding: 3px; }}
QToolButton:hover {{ background: #3a3d41; }}
QToolButton:checked {{ background: {SELECTION}; border-color: {ACCENT}; }}
QTreeView {{ background: {BASE}; alternate-background-color: #232325; border: none; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none;
    border-right: 1px solid {BORDER}; padding: 3px 6px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; margin-top: 16px;
    padding: 8px 6px 4px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {ACCENT}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 5px 10px; border: none;
    border-top: 2px solid transparent; }}
QTabBar::tab:selected {{ background: {BASE}; color: {TEXT}; border-top: 2px solid {ACCENT}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QStatusBar QLabel {{ padding: 0 8px; }}
QToolTip {{ background: {WINDOW}; color: {TEXT}; border: 1px solid #454545; padding: 4px; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    role = QtGui.QPalette.ColorRole
    colors = {
        role.Window: WINDOW, role.WindowText: TEXT, role.Base: BASE, role.AlternateBase: "#232325",
        role.ToolTipBase: WINDOW, role.ToolTipText: TEXT, role.Text: TEXT, role.Button: "#333337",
        role.ButtonText: TEXT, role.BrightText: "#ffffff", role.Highlight: SELECTION,
        role.HighlightedText: "#ffffff", role.Link: ACCENT, role.PlaceholderText: MUTED,
        role.Light: "#3c3c3c", role.Midlight: "#333337", role.Mid: "#2a2a2d", role.Dark: "#1b1b1c",
        role.Shadow: "#000000",
    }
    palette = QtGui.QPalette()
    for r, color in colors.items():
        palette.setColor(r, QtGui.QColor(color))
    for r in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(QtGui.QPalette.ColorGroup.Disabled, r, QtGui.QColor("#6d6d6d"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE_SHEET + _tab_close_style())


def _tab_close_style() -> str:
    """Style sheets take images from files: write the tab close icons to the temp folder."""
    folder = Path(tempfile.gettempdir()) / "lidarlab" / "icons"
    folder.mkdir(parents=True, exist_ok=True)
    rules = []
    for state, color in (("", MUTED), (":hover", TEXT)):
        path = folder / f"close{state.replace(':', '-')}.png"
        icon("close", color).pixmap(16, 16).save(str(path))
        rules.append(f'QTabBar::close-button{state} {{ image: url("{path.as_posix()}"); }}')
    return "\n".join(rules)


# 24 x 24 SVG bodies; {c} is replaced by the icon color
_ICONS = {
    "play": '<path d="M7 4.5v15l12.5-7.5z" fill="{c}"/>',
    "pause": '<rect x="6" y="5" width="4" height="14" rx="1" fill="{c}"/>'
             '<rect x="14" y="5" width="4" height="14" rx="1" fill="{c}"/>',
    "first": '<rect x="4" y="5" width="2.2" height="14" fill="{c}"/>'
             '<path d="M13 5v14L6.6 12zM20 5v14l-6.4-7z" fill="{c}"/>',
    "prev": '<path d="M15.5 5v14L6 12z" fill="{c}"/>'
            '<rect x="16.8" y="5" width="2.2" height="14" fill="{c}"/>',
    "next": '<rect x="5" y="5" width="2.2" height="14" fill="{c}"/>'
            '<path d="M8.5 5v14l9.5-7z" fill="{c}"/>',
    "last": '<path d="M4 5v14l6.4-7zM11 5v14l6.4-7z" fill="{c}"/>'
            '<rect x="17.8" y="5" width="2.2" height="14" fill="{c}"/>',
    "loop": '<path d="M5 12a5 5 0 0 1 5-5h7" fill="none" stroke="{c}" stroke-width="2"/>'
            '<path d="M15.5 3.5 19.5 7l-4 3.5z" fill="{c}"/>'
            '<path d="M19 12a5 5 0 0 1-5 5H7" fill="none" stroke="{c}" stroke-width="2"/>'
            '<path d="M8.5 13.5 4.5 17l4 3.5z" fill="{c}"/>',
    "open": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5'
            ' 1.5h-15A1.5 1.5 0 0 1 3 17.5z" fill="{c}"/>',
    "new": '<path d="M6 3.5h8l4.5 4.5v12.5H6z" fill="none" stroke="{c}" stroke-width="1.6"/>'
           '<path d="M13.5 3.5V8.5h5" fill="none" stroke="{c}" stroke-width="1.6"/>',
    "save": '<path d="M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z" fill="none" stroke="{c}"'
            ' stroke-width="1.6"/><rect x="8" y="4" width="7" height="5" fill="{c}"/>'
            '<rect x="8" y="13" width="8" height="5" fill="{c}"/>',
    "run": '<path d="M7 4.5v15l12.5-7.5z" fill="#4ec96b"/>',
    "run_line": '<path d="M4 5v10l8-5z" fill="#4ec96b"/>'
                '<rect x="13" y="8" width="8" height="2" fill="{c}"/>'
                '<rect x="4" y="14" width="17" height="2" fill="{c}"/>'
                '<rect x="4" y="18" width="12" height="2" fill="{c}"/>',
    "refresh": '<path d="M18.5 9A7 7 0 1 0 19 13.5" fill="none" stroke="{c}" stroke-width="2"/>'
               '<path d="M20 3.5V10h-6.5z" fill="{c}"/>',
    "fit": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" fill="none" stroke="{c}"'
           ' stroke-width="2"/>',
    "add": '<path d="M11 5h2v14h-2zM5 11h14v2H5z" fill="{c}"/>',
    "remove": '<path d="M5 11h14v2H5z" fill="{c}"/>',
    "close": '<path d="M7 7l10 10M17 7 7 17" stroke="{c}" stroke-width="2"/>',
    "up": '<path d="M12 5l6.5 7H14v7h-4v-7H5.5z" fill="{c}"/>',
    "down": '<path d="M12 19l6.5-7H14V5h-4v7H5.5z" fill="{c}"/>',
    "trash": '<path d="M9 4h6l1 2h4v2H4V6h4zM6 9.5h12l-1 10.5H7z" fill="{c}"/>',
    "camera": '<path d="M4 7h3.5L9 5h6l1.5 2H20a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8'
              'a1 1 0 0 1 1-1z" fill="{c}"/><circle cx="12" cy="13" r="3.4" fill="#252526"/>',
    "memory": '<rect x="5" y="6" width="14" height="12" rx="1.5" fill="none" stroke="{c}"'
              ' stroke-width="1.6"/>'
              '<path d="M8 3v3M12 3v3M16 3v3M8 18v3M12 18v3M16 18v3" stroke="{c}"'
              ' stroke-width="1.6"/>'
              '<rect x="9" y="9.5" width="6" height="5" fill="{c}"/>',
    "tape": '<rect x="3" y="6" width="18" height="12" rx="2" fill="none" stroke="{c}"'
            ' stroke-width="1.6"/>'
            '<circle cx="8.5" cy="12" r="2.3" fill="{c}"/>'
            '<circle cx="15.5" cy="12" r="2.3" fill="{c}"/>',
    "folder": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1'
              '-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z" fill="none" stroke="{c}" stroke-width="1.6"/>',
    "split": '<rect x="3" y="5" width="18" height="14" rx="1.5" fill="none" stroke="{c}"'
             ' stroke-width="1.6"/><path d="M12 5v14" stroke="{c}" stroke-width="1.6"/>'
             '<circle cx="7.5" cy="12" r="1.3" fill="{c}"/><circle cx="16.5" cy="12" r="1.3"'
             ' fill="#4ec96b"/>',
    "lidar": '<circle cx="12" cy="12" r="9" fill="none" stroke="{c}" stroke-width="1.6"/>'
             '<circle cx="12" cy="12" r="5.5" fill="none" stroke="{c}" stroke-width="1.6"/>'
             '<circle cx="12" cy="12" r="2.2" fill="{c}"/>',
}


@cache
def icon(name: str, color: str = TEXT) -> QtGui.QIcon:
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
           + _ICONS[name].format(c=color) + "</svg>")
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    result = QtGui.QIcon()
    for size in (16, 20, 24, 32, 48):
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        result.addPixmap(pixmap)
    return result


def app_icon() -> QtGui.QIcon:
    return icon("lidar", ACCENT)


@cache
def dot_icon(color: str) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(16, 16)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawEllipse(QtCore.QRectF(3, 3, 10, 10))
    painter.end()
    return QtGui.QIcon(pixmap)


@cache
def text_icon(text: str, color: str = "#c586c0") -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(16, 16)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    font = QtGui.QFont("Cambria", 12)
    font.setItalic(True)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(color))
    painter.drawText(QtCore.QRectF(0, -1, 16, 17), QtCore.Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    return QtGui.QIcon(pixmap)


@cache
def colormap_icon(name: str, width: int = 64, height: int = 12) -> QtGui.QIcon:
    from matplotlib import colormaps

    rgb = (colormaps[name](np.linspace(0.0, 1.0, width))[:, :3] * 255).astype(np.uint8)
    img = np.ascontiguousarray(np.repeat(rgb[None], height, axis=0))
    image = QtGui.QImage(img.data, width, height, 3 * width, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QIcon(QtGui.QPixmap.fromImage(image.copy()))
