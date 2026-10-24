"""The editor's side of plugins (trafilatura3d/plugins.py): what a plugin may add to the main window, tracked per
plugin so unloading removes it, and the Plugin Manager (Tools > Plugin Manager): every plugin found, its
folder and status, enable / disable, reload, the search paths (the project's, read only, and the user's)."""

from __future__ import annotations

import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QDialog, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QListWidget, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from trafilatura3d import plugins
from trafilatura3d.editor.keys import variants


class EditorAccess(plugins.EditorAccess):
    def __init__(self, window, name: str):
        self.window, self.name = window, name
        self.actions: list[tuple] = []  # (menu, action)
        self.menus: list = []  # menus this plugin created
        self.docks: list = []

    def _menu(self, workspace: str, title: str):
        """A workspace's menu `title` (the window's own, or one this plugin creates)."""
        before = list(self.window._menus_of(workspace))
        menu = self.window.workspace_menu(workspace, title)
        if menu not in before:
            self.menus.append((workspace, menu))
        return menu

    def add_menu_action(self, menu, text, callback, shortcut="", workspaces=("graph", "paint")):
        action = QAction(text, self.window)
        if shortcut:
            action.setShortcuts(variants(shortcut))
            self.window.addAction(action)  # a menu is a window of its own: shortcuts work on the main window
        action.triggered.connect(lambda _checked=False: self._call(callback))
        for workspace in workspaces:  # one action in each workspace's menu of that title
            m = self._menu(workspace, menu)
            m.addAction(action)
            self.actions.append((m, action))
        return action

    def add_menu_separator(self, menu, workspaces=("graph", "paint")):
        for workspace in workspaces:
            m = self._menu(workspace, menu)
            self.actions.append((m, m.addSeparator()))

    def workspace(self):
        return self.window.shown

    def current_paint(self):
        doc = self.window.paint.doc
        return doc.session if doc is not None else None

    def edit(self, package, label, fn):
        self.window.doc.edit(package, label, fn)

    def open(self, path):
        self.window.open_path(path)

    def add_dock(self, title, widget):
        dock = QDockWidget(title, self.window.workspace)
        dock.setObjectName(f"plugin_{self.name}_{len(self.docks)}")
        dock.setWidget(widget)
        self.window.workspace.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.docks.append(dock)
        return dock

    def main_window(self):
        return self.window

    def current(self):
        g = self.window.graph
        return (g.package, g.graph_id) if g.package is not None else (None, "")

    def save(self, document):
        w = self.window
        paint = w.paint.doc
        saved = w.paint.save() if paint is not None and document is paint.session else w.save_package(document)
        if not saved:
            raise OSError(f"{document.path.name if document.path else 'the document'} could not be saved")

    def remove_all(self):
        for menu, action in self.actions:
            menu.removeAction(action)
            self.window.removeAction(action)
        for action in {id(a): a for _, a in self.actions}.values():  # one action can sit in both workspaces' menus
            action.deleteLater()
        for workspace, menu in self.menus:
            if menu.isEmpty():
                self.window.remove_workspace_menu(workspace, menu)
                menu.deleteLater()
        for dock in self.docks:
            self.window.workspace.removeDockWidget(dock)
            dock.deleteLater()
        self.actions, self.menus, self.docks = [], [], []

    def _call(self, callback):
        try:
            callback()
        except plugins.Veto as v:
            self.window.status(str(v))
        except Exception as e:  # noqa: BLE001 - a plugin's error is reported, never fatal
            plugins.manager().report(f"plugin {self.name}: {type(e).__name__}: {e}")


def install(window) -> None:
    """Plugins in the editor: their editor access, messages in the status bar, then load the enabled ones."""
    m = plugins.manager()
    m.editor_factory = lambda name: EditorAccess(window, name)
    m.reporter = window.status
    m.load_all()


STATUS = {True: "Loaded", False: "Not loaded"}


class PluginManager(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plugin Manager")
        self.resize(760, 520)
        self.m = plugins.manager()
        lay = QVBoxLayout(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Plugin", "Status", "Description", "Location"])
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 110)
        self.tree.setColumnWidth(2, 220)
        self.tree.itemChanged.connect(self._toggled)
        self.tree.currentItemChanged.connect(lambda *_: self._show_error())
        lay.addWidget(self.tree, 2)
        self.error = QLabel("")
        self.error.setObjectName("dim")
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.error)
        row = QHBoxLayout()
        reload = QPushButton("Reload")
        reload.setToolTip("Unload and load the selected plugin again (after editing its files)")
        reload.clicked.connect(self._reload)
        folder = QPushButton("Open folder")
        folder.clicked.connect(self._open_folder)
        rescan = QPushButton("Refresh")
        rescan.setToolTip("Look for plugins again in the search paths")
        rescan.clicked.connect(self._rescan)
        for b in (reload, folder, rescan):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel("Search paths (the project's come from the project settings)"))
        self.paths = QListWidget()
        lay.addWidget(self.paths, 1)
        prow = QHBoxLayout()
        add = QPushButton("Add path…")
        add.clicked.connect(self._add_path)
        remove = QPushButton("Remove path")
        remove.clicked.connect(self._remove_path)
        prow.addWidget(add)
        prow.addWidget(remove)
        prow.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        prow.addWidget(close)
        lay.addLayout(prow)
        self._fill()

    def _fill(self):
        self.m.discover()
        self.tree.blockSignals(True)
        self.tree.clear()
        for info in self.m.plugins.values():
            status = "Failed" if info.error else STATUS[info.loaded] if info.enabled else "Disabled"
            it = QTreeWidgetItem([info.name, status, info.description, str(info.path)])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(0, Qt.Checked if info.enabled else Qt.Unchecked)
            it.setToolTip(3, str(info.path))
            it.setData(0, Qt.UserRole, info.name)
            self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        self.paths.clear()
        user = set(self.m.settings()["paths"])
        for p in self.m.search_paths():
            self.paths.addItem(p + ("" if p in user else "   (project / default)"))
            self.paths.item(self.paths.count() - 1).setData(Qt.UserRole, p if p in user else None)
        self._show_error()

    def _selected(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it else None

    def _show_error(self):
        name = self._selected()
        info = self.m.plugins.get(name) if name else None
        self.error.setText(info.error if info and info.error else "")

    def _toggled(self, item, column):
        if column == 0:
            self.m.set_enabled(item.data(0, Qt.UserRole), item.checkState(0) == Qt.Checked)
            self._fill()

    def _reload(self):
        name = self._selected()
        if name:
            self.m.reload(name)
            self._fill()

    def _rescan(self):
        self.m.load_all()
        self._fill()

    def _open_folder(self):
        name = self._selected()
        if name and sys.platform == "darwin":
            info = self.m.plugins[name]
            subprocess.Popen(["open", str(info.path if info.path.is_dir() else info.path.parent)])

    def _add_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Plugin folder")
        if folder:
            data = self.m.settings()
            if folder not in data["paths"]:
                data["paths"].append(folder)
                self.m.save_settings(data)
            self._rescan()

    def _remove_path(self):
        it = self.paths.currentItem()
        p = it.data(Qt.UserRole) if it else None
        if p:  # only the user's paths: the project's are edited in the project settings
            data = self.m.settings()
            data["paths"] = [x for x in data["paths"] if x != p]
            self.m.save_settings(data)
            self._fill()
