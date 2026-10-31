"""Script editor: Python tabs with highlighting, line numbers, completion from the live shell
namespace, run file / run selection, and reload when a file changes on disk.

With "Run on external save" on, saving a script in another editor (e.g. VS Code) reloads the
tab and runs the script again, so its ``@lab.step`` functions replace the old ones.
"""

from __future__ import annotations

import builtins
import keyword
import re
import tempfile
import textwrap
from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .paths import examples_dir

Qt = QtCore.Qt


def _format(color: str, italic: bool = False) -> QtGui.QTextCharFormat:
    fmt = QtGui.QTextCharFormat()
    fmt.setForeground(QtGui.QColor(color))
    fmt.setFontItalic(italic)
    return fmt


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Python colors in the style of VS Code's dark theme."""

    KEYWORDS = [k for k in keyword.kwlist + keyword.softkwlist if k != "_"]
    BUILTINS = [n for n in dir(builtins) if not n.startswith("_")]
    STRING_OR_COMMENT = re.compile(
        r"(?P<comment>#.*)"
        r"|(?P<triple>[rRbBuUfF]{0,2}(?:'''|\"\"\"))"
        r"|(?P<string>[rRbBuUfF]{0,2}(?:'(?:\\.|[^'\\])*'?|\"(?:\\.|[^\"\\])*\"?))"
    )

    def __init__(self, document: QtGui.QTextDocument):
        super().__init__(document)
        self.string = _format("#ce9178")
        self.comment = _format("#6a9955", italic=True)
        self.rules = [
            (re.compile(rf"\b(?:{'|'.join(self.KEYWORDS)})\b"), _format("#569cd6"), 0),
            (re.compile(rf"\b(?:{'|'.join(self.BUILTINS)})\b"), _format("#4ec9b0"), 0),
            (re.compile(r"\bself\b"), _format("#9cdcfe", italic=True), 0),
            (re.compile(r"\b0[xXoObB][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?j?"),
             _format("#b5cea8"), 0),
            (re.compile(r"^\s*@[\w.]+"), _format("#dcdcaa"), 0),
            (re.compile(r"\bdef\s+(\w+)"), _format("#dcdcaa"), 1),
            (re.compile(r"\bclass\s+(\w+)"), _format("#4ec9b0"), 1),
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt, group in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(group), m.end(group) - m.start(group), fmt)
        # strings and comments, left to right, so '#' inside a string is not a comment
        start = 0
        state = self.previousBlockState()
        if state in (1, 2):  # inside a triple-quoted string that began on an earlier line
            end = text.find("'''" if state == 1 else '"""')
            if end < 0:
                self.setFormat(0, len(text), self.string)
                self.setCurrentBlockState(state)
                return
            self.setFormat(0, end + 3, self.string)
            start = end + 3
        self.setCurrentBlockState(0)
        while m := self.STRING_OR_COMMENT.search(text, start):
            if m.group("comment") is not None:
                self.setFormat(m.start(), len(text) - m.start(), self.comment)
                return
            if m.group("triple") is not None:
                delim = m.group("triple")[-3:]
                end = text.find(delim, m.end())
                if end < 0:
                    self.setFormat(m.start(), len(text) - m.start(), self.string)
                    self.setCurrentBlockState(1 if delim == "'''" else 2)
                    return
                self.setFormat(m.start(), end + 3 - m.start(), self.string)
                start = end + 3
                continue
            self.setFormat(m.start(), m.end() - m.start(), self.string)
            start = m.end()


class _LineNumbers(QtWidgets.QWidget):
    def __init__(self, editor: CodeEditor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.editor.line_number_width(), 0)

    def paintEvent(self, event) -> None:
        self.editor.paint_line_numbers(event)


class CodeEditor(QtWidgets.QPlainTextEdit):
    """Python editor with line numbers, auto-indent, comment toggle and jedi completion."""

    def __init__(self, namespace: Callable[[], dict], parent=None):
        super().__init__(parent)
        self.namespace = namespace
        self.path: Path | None = None
        self.suggested_name = "script.py"
        font = QtGui.QFont("Consolas", 10)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self._numbers = _LineNumbers(self)
        self.blockCountChanged.connect(self._update_margin)
        self.updateRequest.connect(self._update_numbers)
        self.cursorPositionChanged.connect(self._highlight_line)
        self._update_margin()
        self._highlight_line()
        self.highlighter = PythonHighlighter(self.document())
        self.completer = QtWidgets.QCompleter(self)
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setModel(QtCore.QStringListModel(self.completer))
        self.completer.activated.connect(self._insert_completion)

    # line numbers and current line

    def line_number_width(self) -> int:
        digits = max(3, len(str(self.blockCount())))
        return 14 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_margin(self, *_) -> None:
        self.setViewportMargins(self.line_number_width(), 0, 0, 0)

    def _update_numbers(self, rect: QtCore.QRect, dy: int) -> None:
        if dy:
            self._numbers.scroll(0, dy)
        else:
            self._numbers.update(0, rect.y(), self._numbers.width(), rect.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        r = self.contentsRect()
        width = self.line_number_width()
        self._numbers.setGeometry(QtCore.QRect(r.left(), r.top(), width, r.height()))

    def paint_line_numbers(self, event) -> None:
        painter = QtGui.QPainter(self._numbers)
        painter.fillRect(event.rect(), QtGui.QColor(theme.BASE))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        width, height = self._numbers.width() - 8, self.fontMetrics().height()
        right = Qt.AlignmentFlag.AlignRight
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QtGui.QColor("#c6c6c6" if number == current else "#6e7681"))
                painter.drawText(0, top, width, height, right, str(number + 1))
            block = block.next()
            top, bottom = bottom, bottom + round(self.blockBoundingRect(block).height())
            number += 1
        painter.end()

    def _highlight_line(self) -> None:
        selection = QtWidgets.QTextEdit.ExtraSelection()
        selection.format.setBackground(QtGui.QColor("#2a2d2e"))
        selection.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        self.setExtraSelections([selection])

    def goto_line(self, line: int) -> None:
        block = self.document().findBlockByNumber(max(line - 1, 0))
        cursor = QtGui.QTextCursor(block)
        self.setTextCursor(cursor)
        self.centerCursor()

    # editing

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        popup = self.completer.popup()
        key, mods = event.key(), event.modifiers()
        if popup.isVisible() and key in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Tab,
                                         Qt.Key.Key_Backtab, Qt.Key.Key_Escape):
            event.ignore()  # the completer handles these
            return
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if key == Qt.Key.Key_Space and ctrl:
            self.complete()
        elif key == Qt.Key.Key_Slash and ctrl:
            self.toggle_comment()
        elif key == Qt.Key.Key_Tab and not mods:
            self._indent()
        elif key == Qt.Key.Key_Backtab:
            self._dedent()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not mods:
            self._newline()
        else:
            super().keyPressEvent(event)
            text = event.text()
            if text == ".":
                before = self.textCursor().block().text()[: self.textCursor().positionInBlock() - 1]
                if before and (before[-1].isalnum() or before[-1] in "_)]"):
                    self.complete(auto=True)
            elif popup.isVisible():
                if text and (text.isalnum() or text == "_"):
                    self._refresh_prefix()
                elif key != Qt.Key.Key_Backspace:
                    popup.hide()
                else:
                    self._refresh_prefix()

    def _newline(self) -> None:
        cursor = self.textCursor()
        line = cursor.block().text()[: cursor.positionInBlock()]
        indent = re.match(r"\s*", line).group(0)
        if line.rstrip().endswith(":"):
            indent += "    "
        cursor.insertText("\n" + indent)
        self.ensureCursorVisible()

    def _blocks(self) -> tuple[int, int]:
        cursor = self.textCursor()
        doc = self.document()
        first = doc.findBlock(cursor.selectionStart()).blockNumber()
        end_block = doc.findBlock(cursor.selectionEnd())
        last = end_block.blockNumber()
        if cursor.hasSelection() and last > first and end_block.position() == cursor.selectionEnd():
            last -= 1  # selection ends at the start of a line: that line is not included
        return first, last

    def _edit_lines(self, change: Callable[[str], str]) -> None:
        first, last = self._blocks()
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for number in range(first, last + 1):
            block = self.document().findBlockByNumber(number)
            new = change(block.text())
            if new != block.text():
                c = QtGui.QTextCursor(block)
                c.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock,
                               QtGui.QTextCursor.MoveMode.KeepAnchor)
                c.insertText(new)
        cursor.endEditBlock()

    def _indent(self) -> None:
        if self.textCursor().hasSelection() and self._blocks()[1] > self._blocks()[0]:
            self._edit_lines(lambda line: "    " + line if line.strip() else line)
        else:
            self.textCursor().insertText("    ")

    def _dedent(self) -> None:
        self._edit_lines(lambda line: line[min(4, len(line) - len(line.lstrip(" "))):])

    def toggle_comment(self) -> None:
        first, last = self._blocks()
        lines = [self.document().findBlockByNumber(n).text() for n in range(first, last + 1)]
        code = [line for line in lines if line.strip()]
        if not code:
            return
        if all(line.lstrip().startswith("#") for line in code):
            self._edit_lines(lambda line: re.sub(r"^(\s*)# ?", r"\1", line, count=1))
        else:
            indent = min(len(line) - len(line.lstrip()) for line in code)

            def comment(line: str) -> str:
                return line[:indent] + "# " + line[indent:] if line.strip() else line

            self._edit_lines(comment)

    # completion

    def complete(self, auto: bool = False) -> None:
        try:
            import jedi

            cursor = self.textCursor()
            script = jedi.Interpreter(self.toPlainText(), [self.namespace()])
            completions = script.complete(cursor.blockNumber() + 1, cursor.positionInBlock())
        except Exception:  # noqa: BLE001 - completion is best effort
            return
        names = [c.name for c in completions if not (auto and c.name.startswith("_"))][:500]
        if not names:
            self.completer.popup().hide()
            return
        self.completer.model().setStringList(names)
        self._refresh_prefix()
        rect = self.cursorRect()
        popup = self.completer.popup()
        scrollbar = popup.verticalScrollBar().sizeHint().width()
        rect.setWidth(popup.sizeHintForColumn(0) + scrollbar + 20)
        self.completer.complete(rect)

    def _word_before_cursor(self) -> str:
        cursor = self.textCursor()
        m = re.search(r"[A-Za-z_]\w*$", cursor.block().text()[: cursor.positionInBlock()])
        return m.group(0) if m else ""

    def _refresh_prefix(self) -> None:
        self.completer.setCompletionPrefix(self._word_before_cursor())
        if self.completer.completionCount() == 0:
            self.completer.popup().hide()
        else:
            self.completer.popup().setCurrentIndex(self.completer.completionModel().index(0, 0))

    def _insert_completion(self, name: str) -> None:
        cursor = self.textCursor()
        prefix = self._word_before_cursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left,
                            QtGui.QTextCursor.MoveMode.KeepAnchor, len(prefix))
        cursor.insertText(name)
        self.setTextCursor(cursor)


class ScriptEditor(QtWidgets.QWidget):
    """Tabs of scripts with a toolbar; asks the shell to run them through signals."""

    runFile = QtCore.Signal(str)
    runCode = QtCore.Signal(str)

    def __init__(self, namespace: Callable[[], dict], scripts_dir: Path, parent=None):
        super().__init__(parent)
        self.namespace = namespace
        self.scripts_dir = scripts_dir
        self._untitled = 0
        self._tmp = Path(tempfile.gettempdir()) / "lidarlab"
        self.watcher = QtCore.QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._file_changed)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)

        def action(text, slot, shortcut=None, icon=None, context=Qt.ShortcutContext.WindowShortcut):
            a = QtGui.QAction(text, self)
            if icon:
                a.setIcon(theme.icon(icon))
            if shortcut:
                a.setShortcut(QtGui.QKeySequence(shortcut))
                a.setShortcutContext(context)
            a.triggered.connect(slot)
            self.addAction(a)
            return a

        local = Qt.ShortcutContext.WidgetWithChildrenShortcut
        self.actions_ = {
            "new": action("&New Script", lambda: self.new_tab(), "Ctrl+N", "new"),
            "open": action("&Open Script…", lambda: self.open_file(), "Ctrl+Shift+O", "open"),
            "save": action("&Save Script", lambda: self.save(), "Ctrl+S", "save", local),
            "save_as": action("Save Script &As…", lambda: self.save(save_as=True)),
            "run": action("&Run Script", self.run_current, "F5", "run"),
            "run_selection": action("Run &Selection or Line", self.run_selection, "Ctrl+Return",
                                    "run_line", local),
        }
        self.auto_run = QtGui.QAction("Run on external save", self)
        self.auto_run.setCheckable(True)
        self.auto_run.setChecked(True)
        self.auto_run.setToolTip("When a script open here is saved by another editor "
                                 "(e.g. VS Code), reload it and run it again")
        self.examples_menu = QtWidgets.QMenu("Examples", self)
        for path in sorted(examples_dir().glob("*.py")):
            self.examples_menu.addAction(path.name, lambda p=path: self.open_example(p))

        toolbar = QtWidgets.QToolBar()
        toolbar.setIconSize(QtCore.QSize(16, 16))
        for name in ("new", "open", "save"):
            toolbar.addAction(self.actions_[name])
        toolbar.addSeparator()
        for name in ("run", "run_selection"):
            toolbar.addAction(self.actions_[name])
        toolbar.addSeparator()
        toolbar.addAction(self.auto_run)
        examples = QtWidgets.QToolButton()
        examples.setText("Examples")
        examples.setMenu(self.examples_menu)
        examples.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        toolbar.addWidget(examples)
        for name, tip in (("run", "Run the script in the Python shell (F5)"),
                          ("run_selection", "Run the selection or the current line (Ctrl+Enter)")):
            self.actions_[name].setToolTip(tip)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.tabs)

    # tabs

    def current(self) -> CodeEditor | None:
        return self.tabs.currentWidget()

    def editors(self) -> list[CodeEditor]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def new_tab(self, text: str = "", path: Path | None = None,
                title: str | None = None) -> CodeEditor:
        editor = CodeEditor(self.namespace)
        editor.setPlainText(text)
        editor.path = path
        if path is None:
            self._untitled += 1
            editor.suggested_name = title or f"untitled-{self._untitled}.py"
        editor.document().setModified(False)
        editor.document().modificationChanged.connect(lambda _m, e=editor: self._update_title(e))
        index = self.tabs.addTab(editor, "")
        self.tabs.setCurrentIndex(index)
        self._update_title(editor)
        if path is not None:
            self.watcher.addPath(str(path))
        editor.setFocus()
        return editor

    def _update_title(self, editor: CodeEditor) -> None:
        index = self.tabs.indexOf(editor)
        name = editor.path.name if editor.path else editor.suggested_name
        self.tabs.setTabText(index, name + (" •" if editor.document().isModified() else ""))
        self.tabs.setTabToolTip(index, str(editor.path) if editor.path else "not saved yet")

    def open_file(self, path=None, line: int | None = None) -> CodeEditor | None:
        if path is None:
            start = str(self.scripts_dir if self.scripts_dir.is_dir() else Path.cwd())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open script", start,
                                                            "Python (*.py)")
            if not path:
                return None
        path = Path(path).resolve()
        editor = next((e for e in self.editors() if e.path == path), None)
        if editor is None:
            editor = self.new_tab(path.read_text(encoding="utf-8"), path=path)
        self.tabs.setCurrentWidget(editor)
        if line:
            editor.goto_line(line)
        return editor

    def open_example(self, path: Path) -> CodeEditor:
        """Open a copy of an example; saving asks where to put it."""
        return self.new_tab(path.read_text(encoding="utf-8"), title=path.name)

    def close_tab(self, index: int) -> None:
        editor = self.tabs.widget(index)
        if editor.document().isModified():
            button = QtWidgets.QMessageBox.StandardButton
            answer = QtWidgets.QMessageBox.question(
                self, "Close script", f"Save changes to {self.tabs.tabText(index).rstrip(' •')}?",
                button.Save | button.Discard | button.Cancel)
            if answer == button.Cancel:
                return
            if answer == button.Save and not self.save(editor):
                return
        if editor.path is not None:
            self.watcher.removePath(str(editor.path))
        self.tabs.removeTab(index)
        editor.deleteLater()

    def save(self, editor: CodeEditor | None = None, save_as: bool = False) -> bool:
        editor = editor or self.current()
        if editor is None:
            return False
        path = editor.path
        if path is None or save_as:
            self.scripts_dir.mkdir(parents=True, exist_ok=True)
            suggested = str(path or self.scripts_dir / editor.suggested_name)
            name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save script", suggested,
                                                            "Python (*.py)")
            if not name:
                return False
            if editor.path is not None:
                self.watcher.removePath(str(editor.path))
            path = Path(name).resolve()
        path.write_text(editor.toPlainText(), encoding="utf-8")
        editor.path = path
        editor.document().setModified(False)
        self._update_title(editor)
        if str(path) not in self.watcher.files():
            self.watcher.addPath(str(path))
        return True

    def maybe_save_all(self) -> bool:
        """Ask about unsaved scripts before the application closes; False = cancel closing."""
        modified = [e for e in self.editors()
                    if e.document().isModified() and e.toPlainText().strip()]
        if not modified:
            return True
        names = ", ".join(e.path.name if e.path else e.suggested_name for e in modified)
        button = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(self, "Unsaved scripts",
                                                f"Save changes to {names}?",
                                                button.SaveAll | button.Discard | button.Cancel)
        if answer == button.Cancel:
            return False
        if answer == button.SaveAll:
            return all(self.save(e) for e in modified)
        return True

    def save_session(self, settings: QtCore.QSettings) -> None:
        settings.setValue("editor/files", [str(e.path) for e in self.editors() if e.path])

    def restore_session(self, settings: QtCore.QSettings) -> None:
        files = settings.value("editor/files") or []
        for name in [files] if isinstance(files, str) else files:
            if Path(name).is_file():
                self.open_file(name)

    # running

    def run_current(self) -> None:
        editor = self.current()
        if editor is None:
            return
        if editor.path is not None:
            if editor.document().isModified():
                self.save(editor)
            self.runFile.emit(str(editor.path))
            return
        # an unsaved buffer runs from a temporary copy, so tracebacks still show line numbers
        self._tmp.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp / re.sub(r"[^\w.-]", "_", editor.suggested_name)
        tmp.write_text(editor.toPlainText(), encoding="utf-8")
        self.runFile.emit(str(tmp))

    def run_selection(self) -> None:
        editor = self.current()
        if editor is None:
            return
        cursor = editor.textCursor()
        if cursor.hasSelection():
            code = cursor.selection().toPlainText()
        else:
            code = cursor.block().text()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextBlock)
            editor.setTextCursor(cursor)
        code = textwrap.dedent(code)
        if code.strip():
            self.runCode.emit(code)

    # files changed by other programs

    def _file_changed(self, name: str) -> None:
        path = Path(name)
        if not path.exists():
            # editors that save via rename briefly remove the file: look again shortly
            QtCore.QTimer.singleShot(300, lambda: path.exists() and self._file_changed(name))
            return
        if name not in self.watcher.files():
            self.watcher.addPath(name)
        editor = next((e for e in self.editors() if e.path == path.resolve()), None)
        if editor is None:
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return
        if text == editor.toPlainText():
            return  # our own save
        if editor.document().isModified():
            answer = QtWidgets.QMessageBox.question(
                self, "Script changed on disk",
                f"{path.name} was changed by another program. Reload it and lose your edits here?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        line = editor.textCursor().blockNumber() + 1
        scroll = editor.verticalScrollBar().value()
        editor.setPlainText(text)
        editor.document().setModified(False)
        editor.goto_line(line)
        editor.verticalScrollBar().setValue(scroll)
        if self.auto_run.isChecked():
            self.runFile.emit(str(path))
