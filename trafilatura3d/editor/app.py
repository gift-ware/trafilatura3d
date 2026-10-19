"""`trafilatura3d-editor [package.trafilatura3d ...]`: the editor."""

import sys

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication

from trafilatura3d.editor import theme


class _FileOpen(QObject):
    """Documents the system asks the editor to open (a double click in the Finder): opened as a drop opens them,
    kept until the main window exists."""

    def __init__(self):
        super().__init__()
        self._window, self.pending = None, []

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window
        for path in self.pending:
            window.open_path(path)
        self.pending = []

    def eventFilter(self, obj, event):
        if event.type() != QEvent.FileOpen or not event.file():
            return False
        if self._window is None:
            self.pending.append(event.file())
        else:
            self._window.open_path(event.file())
        return True


def main(argv=None) -> int:
    if argv is None:  # the editor started (not a test): the allocator's large block cache off
        from trafilatura3d.editor.macos import without_large_cache
        without_large_cache()
    argv = sys.argv if argv is None else argv
    from PySide6.QtCore import QSettings
    if QSettings("Trafilatura 3D", "Trafilatura 3D").value("preferences/disable_high_dpi", False) in (True, "true"):
        import os  # Preferences > Disable High DPI (read before the application exists)
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    from trafilatura3d.editor.macos import set_app_name
    set_app_name("Trafilatura 3D")  # the menu bar and the Dock say Trafilatura 3D, not Python (before Qt starts Cocoa)
    from trafilatura3d.editor.windows import set_app_id
    set_app_id()  # Windows: a taskbar button of its own with the window's icon, not python.exe's
    app = QApplication(argv)
    app.setApplicationName("Trafilatura 3D")
    app.setOrganizationName("Trafilatura 3D")
    from trafilatura3d.editor.icons import app_icon
    app.setWindowIcon(app_icon())  # also the Dock icon on macOS
    app.setStyle("Fusion")
    theme.load_font()
    app.setPalette(theme.palette())
    app.setStyleSheet(theme.STYLESHEET + theme.dropdown_arrows())
    opened = _FileOpen()  # macOS hands a double-clicked document over as an event, not as an argument
    app.installEventFilter(opened)
    from trafilatura3d.editor import splash
    splash.show()  # until the main window is up; MainWindow reports its steps
    splash.step("Loading the modules")

    # rendercanvas adopts the existing QApplication; with an app made before it is imported its
    # loop.run() returns at once, so the Qt loop is run here (supported by rendercanvas).
    from trafilatura3d import channel
    from trafilatura3d.editor.mainwindow import MainWindow
    window = MainWindow(argv[1:], channel=channel.name())  # agents reach the editor through it (docs/MCP.md)
    window.show()
    opened.window = window  # the documents asked for while starting open now, later ones at once
    splash.follow(window, [window.doc.live, window.paint.live], window.doc.ev)  # until the first opening is done
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
