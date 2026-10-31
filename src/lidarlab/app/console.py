"""Python shell dock: an IPython console whose kernel runs inside the application.

Code typed here (or run from the script editor) sees the same objects the viewer draws:
no copies, no serialisation.
"""

from __future__ import annotations

from PySide6 import QtWidgets

BANNER = """LidarLab Python shell: IPython in the viewer's process
  rec    test run         rec[42], len(rec), rec.rate_hz
  frame  current frame    frame.intensity, frame[frame.echo != 0]
  view   the viewer       view.goto(120), view.output
  pipe   processing steps @lab.step def f(frame): ...
F1: scripting reference. Editor: F5 run, Ctrl+Enter line.

"""


class PythonShell(QtWidgets.QWidget):
    def __init__(self, namespace: dict, parent=None):
        super().__init__(parent)
        from qtconsole.inprocess import QtInProcessKernelManager
        from qtconsole.rich_jupyter_widget import RichJupyterWidget

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel(show_banner=False)
        self.kernel_manager.kernel.gui = "qt"
        self.client = self.kernel_manager.client()
        self.client.start_channels()
        self.widget = RichJupyterWidget(self, font_family="Consolas", font_size=10)
        self.widget.banner = BANNER
        self.widget.set_default_style("linux")
        self.widget.kernel_manager = self.kernel_manager
        self.widget.kernel_client = self.client
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)
        self.shell = self.kernel_manager.kernel.shell
        self.shell.push(namespace)

    @property
    def namespace(self) -> dict:
        return self.shell.user_ns

    def push(self, **values) -> None:
        self.shell.user_ns.update(values)

    def execute(self, code: str) -> None:
        """Run code as if typed at the prompt (it is shown and kept in the history)."""
        self.widget.execute(code)

    def run_file(self, path: str) -> None:
        """Run a script in the shell's namespace (IPython's ``%run -i``)."""
        self.widget.execute(f'%run -i "{path}"')

    def print_error(self, text: str) -> None:
        self.widget._append_plain_text(f"\x1b[31m{text}\x1b[0m\n", before_prompt=True)

    def shutdown(self) -> None:
        self.client.stop_channels()
        self.kernel_manager.shutdown_kernel()
