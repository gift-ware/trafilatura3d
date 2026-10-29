"""The Autosave plugin (Autosave plugin page): periodic copies of the open project, which itself is not saved. Plugins > Autosave > Disable / Configure; before a save, for the warning time, a progress bar and a
Snooze button (Snooze delays the save by the snooze interval).

The copies are written into the .autosave folder next to the project file, as the Graph's packages' (`backups`); an
untitled project's go to the user's Documents folder (`Trafilatura 3D/autosave`, ours), or to the folder of "Always save
in the following directory". The timer does not run while a modal window is open (the configuration, the export) and
skips a save when nothing changed since the last backup (ours).

Ours, the documentation gives none: the defaults (10 min, 3 files, 5 min snooze, 10 s warning) and the file
names, `<project>_<date>_<time>.trafilatura3dpaint` (the oldest beyond the number kept is removed).
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QSpinBox,
                               QToolButton, QWidget)

from trafilatura3d.editor import backups
from trafilatura3d.paint.session import PROJECT_SUFFIX

KEY = "paint/autosave/"
DEFAULTS = {"enabled": True, "interval": 10.0, "files": 3, "snooze": 5.0, "warning": 10.0, "always": False,
            "directory": ""}


def setting(name):
    v = QSettings().value(KEY + name, DEFAULTS[name])
    d = DEFAULTS[name]
    if isinstance(d, bool):
        return v in (True, "true", 1, "1")
    return type(d)(v)


def set_setting(name, value):
    QSettings().setValue(KEY + name, value)


def documents_folder() -> Path:
    return Path(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or Path.home()) / "Trafilatura 3D" / "autosave"


def backup_folder(project_path) -> Path:
    """The .autosave folder next to the project (`backups`, as the Graph's packages); the directory of "Always save
    in the following directory", or the Documents one for an untitled project: folders of their own."""
    if setting("always") and setting("directory"):
        return Path(setting("directory"))
    return Path(project_path).parent / backups.FOLDER if project_path else documents_folder()


def backup_path(project_path, mesh: str) -> Path:
    """The file the next backup goes to (`backups.backup_path`: the last `files` kept)."""
    stem = Path(project_path).stem if project_path else (Path(mesh).stem or "untitled")
    return backups.backup_path(backup_folder(project_path), stem, PROJECT_SUFFIX, setting("files"))


class Autosave(QWidget):
    """The timer and the warning bar (progress + Snooze) under the viewport."""

    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 2, 6, 2)
        self.label = QLabel()
        row.addWidget(self.label)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(8)
        row.addWidget(self.bar, 1)
        snooze = QToolButton()
        snooze.setText("Snooze")
        snooze.clicked.connect(self.snooze)
        row.addWidget(snooze)
        self.hide()
        self.due = 0.0  # time of the next save
        self._saved_at = None  # (document, undo index) of the last backup
        self.tick = QTimer(self, interval=250, timeout=self._tick)
        self.restart()

    def restart(self):
        self.due = time.monotonic() + setting("interval") * 60.0
        self.hide()
        if setting("enabled"):
            self.tick.start()
        else:
            self.tick.stop()

    def snooze(self):
        self.due = time.monotonic() + setting("snooze") * 60.0
        self.hide()

    def _tick(self):
        if QApplication.activeModalWidget() is not None:  # paused while a modal window is open
            self.due += self.tick.interval() / 1000.0
            return
        left = self.due - time.monotonic()
        warning = setting("warning")
        doc = self.ws.doc
        if doc is None or not self._changed(doc):
            self.hide()
            if left <= 0:
                self.restart()
            return
        if 0 < left <= warning:
            self.label.setText(f"Autosave in {int(left) + 1} s")
            self.bar.setRange(0, int(warning * 1000))
            self.bar.setValue(int((warning - left) * 1000))
            self.show()
            return
        if left <= 0:
            self.save_now()
            self.restart()

    def _changed(self, doc) -> bool:
        return self._saved_at != (id(doc), doc.undo.index())

    def save_now(self) -> Path | None:
        """Write a backup of the open project (its own path is left as it is)."""
        doc = self.ws.doc
        if doc is None:
            return None
        s = doc.session
        path = backup_path(s.path, doc.project.mesh)
        keep = s.path
        try:
            s.save(path)
        except OSError as e:
            self.ws.log(f"Autosave failed: {e}", error=True)
            return None
        finally:
            s.path = keep
        self._saved_at = (id(doc), doc.undo.index())
        self.ws.log(f"Autosave: {path}")
        return path

    def set_enabled(self, on: bool):
        set_setting("enabled", on)
        self.restart()

    def configure(self):
        dlg = QDialog(self.ws.main)
        dlg.setWindowTitle("Autosave configuration")
        form = QFormLayout(dlg)
        interval = QDoubleSpinBox(minimum=0.5, maximum=240.0, value=setting("interval"), suffix=" min")
        files = QSpinBox(minimum=1, maximum=100, value=setting("files"))
        snooze = QDoubleSpinBox(minimum=0.5, maximum=240.0, value=setting("snooze"), suffix=" min")
        warning = QDoubleSpinBox(minimum=0.0, maximum=600.0, value=setting("warning"), suffix=" s")
        always = QCheckBox("Always save in the following directory")
        always.setChecked(setting("always"))
        folder = QLineEdit(setting("directory") or str(documents_folder()))
        browse = QPushButton("…")
        browse.clicked.connect(lambda: (lambda d: d and folder.setText(d))(
            QFileDialog.getExistingDirectory(dlg, "Autosave directory", folder.text())))
        path_row = QHBoxLayout()
        path_row.addWidget(folder, 1)
        path_row.addWidget(browse)
        form.addRow("Autosave interval in minutes", interval)
        form.addRow("Number of autosave files", files)
        form.addRow("Snooze interval in minutes", snooze)
        form.addRow("Warning time before save in seconds", warning)
        form.addRow(always)
        form.addRow(path_row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec():
            for name, value in (("interval", interval.value()), ("files", files.value()), ("snooze", snooze.value()),
                                ("warning", warning.value()), ("always", always.isChecked()),
                                ("directory", folder.text())):
                set_setting(name, value)
            self.restart()
