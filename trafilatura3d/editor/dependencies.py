"""The Dependency Manager (Explorer > package > Dependency manager): every file the package depends
on, its status, and Relocate… to repair a missing one."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QMenu, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from trafilatura3d.diagnostics import dependencies, relocate
from trafilatura3d.editor.icons import icon
from trafilatura3d.package import SUFFIX

COLUMNS = ("Type", "Name", "Path", "Status", "Used by")


class DependencyManager(QDialog):
    def __init__(self, document, package, parent=None):
        super().__init__(parent)
        self.doc, self.pkg = document, package
        self.setWindowTitle(f"Dependency manager - {document.name(package)}")
        self.resize(900, 420)
        lay = QVBoxLayout(self)
        self.info = QLabel()
        self.info.setObjectName("dim")
        lay.addWidget(self.info)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        relocate_b = QPushButton("Relocate…")
        relocate_b.clicked.connect(lambda: self._relocate(self.tree.currentItem()))
        reload_b = QPushButton("Reload package")
        reload_b.clicked.connect(self._reload)
        row.addWidget(relocate_b)
        row.addWidget(reload_b)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        lay.addLayout(row)
        self.fill()

    def fill(self):
        self.tree.clear()
        deps = dependencies(self.pkg)
        missing = 0
        for d in deps:
            users = ", ".join(f"{g}/{n}" for g, n in d.users) or "package resource"
            it = QTreeWidgetItem([d.kind, d.name, d.path, "OK" if d.ok else "Missing", users])
            it.setToolTip(2, d.resolved)
            it.setData(0, Qt.UserRole, d)
            if not d.ok:
                missing += 1
                it.setIcon(3, icon("warning"))
                it.setForeground(3, QColor("#e8a33a"))
            self.tree.addTopLevelItem(it)
        for c in range(len(COLUMNS)):
            self.tree.resizeColumnToContents(c)
        self.info.setText(f"{len(deps)} dependencies, {missing} missing" if deps else "This package has no dependencies.")

    def _menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None:
            return
        m = QMenu(self)
        m.addAction("Relocate…", lambda: self._relocate(it))
        d = it.data(0, Qt.UserRole)
        if d.ok:
            m.addAction("Show in folder", lambda: _reveal(d.resolved))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _relocate(self, it):
        if it is None:
            return
        d = it.data(0, Qt.UserRole)
        start = str(Path(d.resolved).parent) if Path(d.resolved).parent.exists() else ""
        filt = {"Package": f"Trafilatura 3D packages (*{SUFFIX})", "Font": "Fonts (*.ttf *.otf *.ttc)",
                "SVG": "SVG (*.svg)"}.get(d.kind, "All files (*)")
        path, _ = QFileDialog.getOpenFileName(self, f"Relocate {d.name}", start, filt)
        if not path:
            return
        self.doc.edit(self.pkg, "Relocate dependency", lambda: relocate(self.pkg, d.path, path))
        self.doc.check(self.pkg)  # the files it found: their warnings gone
        self.fill()

    def _reload(self):
        reload_package(self.doc, self.pkg, self)
        self.fill()


def reload_package(document, package, parent=None):
    """Read the package again from disk (Reload), keeping it open in the same place."""
    from PySide6.QtWidgets import QMessageBox
    from trafilatura3d.package import DocumentError, Package
    if package.path is None:
        QMessageBox.information(parent, "Reload", "The package was never saved.")
        return
    if id(package) in document.dirty and QMessageBox.question(
            parent, "Reload", "Discard the unsaved changes and reload from disk?") != QMessageBox.Yes:
        return
    try:
        fresh = Package.load(package.path)
    except (DocumentError, OSError, ValueError) as e:
        document.error.emit(f"Reload: {e}")
        return
    package.restore(fresh.to_dict())
    package._external.clear()
    document.dirty.discard(id(package))
    document.undo.clear()
    document.check(package)  # loaded again: what is wrong in it, as when it opens
    document.changed.emit(package)
    document.packages_changed.emit()


def _reveal(path):
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", path])
    else:
        subprocess.Popen(["xdg-open", str(Path(path).parent)])
