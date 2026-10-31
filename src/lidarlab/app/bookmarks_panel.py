"""Bookmarks dock: notes on frames of the open test run; double-click goes to the frame."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from lidarlab.core.bookmarks import Bookmarks

from . import theme
from .timeline import format_time

FRAME_ROLE = QtCore.Qt.ItemDataRole.UserRole


class BookmarksPanel(QtWidgets.QWidget):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.w = window
        self.bookmarks: Bookmarks | None = None
        self._building = False
        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.addAction(theme.icon("add"), "Bookmark the current frame (B)",
                          lambda: window.add_bookmark())
        toolbar.addAction(theme.icon("remove"), "Remove the selected bookmark", self._remove)
        toolbar.addSeparator()
        toolbar.addAction(theme.icon("prev"), "Previous bookmark ([)",
                          lambda: window.jump_bookmark(-1))
        toolbar.addAction(theme.icon("next"), "Next bookmark (])", lambda: window.jump_bookmark(1))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Frame", "Time", "Note"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        header = self.tree.header()
        header.resizeSection(0, 56)
        header.resizeSection(1, 64)
        header.setStretchLastSection(True)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
                                  | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.itemChanged.connect(self._item_changed)
        self.empty = QtWidgets.QLabel("No bookmarks. Press B on a frame to add one.")
        self.empty.setStyleSheet(f"color: {theme.MUTED}; padding: 6px;")
        self.empty.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.tree)
        layout.addWidget(self.empty)

    def set_bookmarks(self, bookmarks: Bookmarks | None) -> None:
        self.bookmarks = bookmarks
        self.rebuild()

    def rebuild(self) -> None:
        self._building = True
        self.tree.clear()
        b = self.bookmarks
        rate = self.w.rec.rate_hz if self.w.rec is not None else 10.0
        for frame in (b.frames() if b is not None else []):
            item = QtWidgets.QTreeWidgetItem([str(frame), format_time(frame / rate), b.note(frame)])
            item.setData(0, FRAME_ROLE, frame)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(2, "Click again or press F2 to edit the note")
            self.tree.addTopLevelItem(item)
        self.empty.setVisible(not self.tree.topLevelItemCount())
        self._building = False

    def _double_clicked(self, item, column: int) -> None:
        if column == 2:
            self.tree.editItem(item, 2)
        else:
            self.w.show_frame(item.data(0, FRAME_ROLE), wait=True)

    def _item_changed(self, item, column: int) -> None:
        if self._building or column != 2 or self.bookmarks is None:
            return
        self.bookmarks.add(item.data(0, FRAME_ROLE), item.text(2))
        self.w.save_bookmarks()

    def _remove(self) -> None:
        item = self.tree.currentItem()
        if item is not None and self.bookmarks is not None:
            self.bookmarks.remove(item.data(0, FRAME_ROLE))
            self.w.save_bookmarks()
