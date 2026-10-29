"""The main window. Two workspaces, chosen by the type of the open document (decision 23): Graph, laid out
with Explorer and Library on the left, the Graph in the middle with the 3D and 2D views
below it, Properties on the right, and Paint (trafilatura3d/editor/paint/workspace.py), each with its own main
menu and shortcuts. Every Graph panel is a dock with a title bar ("Name · Panel", pin / undock /
maximize / close); Shift+Space maximizes the dock under the mouse; Windows > Reset layout restores the default.
The Home screen shows while nothing is open."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMenu, QMessageBox, QStackedWidget, QToolBar, QToolButton, QWidget)

from trafilatura3d import __version__
from trafilatura3d.editor import splash, theme
from trafilatura3d.editor.document import Document
from trafilatura3d.editor.explorer import ExplorerPanel
from trafilatura3d.editor.keys import variants
from trafilatura3d.editor.graph.panel import GraphPanel
from trafilatura3d.editor.home import HomeScreen, NewGraphDialog
from trafilatura3d.editor.icons import icon
from trafilatura3d.editor.library import LibraryPanel
from trafilatura3d.editor.live import LiveCook, Plan
from trafilatura3d.editor.properties import PropertiesPanel
from trafilatura3d.editor.view2d import View2D
from trafilatura3d.editor.view3d import View3D
from trafilatura3d.package import SUFFIX, DocumentError, apply_template

VERSION = __version__  # About, the status bar
LAYOUT_KEY = "workspace/state/2"  # bump when the default layout changes, so old saved layouts are dropped


class DockTitle(QWidget):
    """The dock title bar: icon, "Name · Panel", then pin, undock, maximize and close. An accent line under it marks
    the panel with the keyboard focus."""

    def __init__(self, dock: QDockWidget, icon_name: str, window):
        super().__init__()
        self.dock, self.window = dock, window
        self.setObjectName("dockTitle")
        self.active = None
        self._set_active(False)
        theme.divider(self, "bottom")  # the title meets the panel's body
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 4, 3)
        lay.setSpacing(2)
        ic = QLabel()
        ic.setPixmap(icon(icon_name).pixmap(16, 16))
        lay.addWidget(ic)
        self.title = QLabel(dock.windowTitle())
        self.title.setStyleSheet("font-weight: 600; padding-left: 4px;")
        self.title.setMinimumWidth(1)  # a long title is cut, it does not widen the panel
        lay.addWidget(self.title, 1)
        self.pin = self._button("pin", "Pin: new content opens in another view", checkable=True)
        self._button("undock", "Undock", lambda: dock.setFloating(not dock.isFloating()))
        self._button("maximize", "Maximize (Shift+Space)", lambda: window.toggle_maximize(dock))
        self._button("close", "Close", dock.close)
        QApplication.instance().focusChanged.connect(self._focus_changed)

    def _button(self, name, tip, slot=None, checkable=False):
        b = QToolButton()
        b.setIcon(icon(name))
        b.setIconSize(QSize(14, 14))
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.setCheckable(checkable)
        if slot:
            b.clicked.connect(slot)
        self.layout().addWidget(b)
        return b

    def set_title(self, text: str):
        self.title.setText(text.replace(" - ", " · "))

    def _focus_changed(self, old, new):
        if new is not None:  # the application losing the focus keeps the active panel
            self._set_active(self.dock.isAncestorOf(new))

    def _set_active(self, on: bool):
        if on != self.active:
            self.active = on
            line = f"2px solid {theme.ACCENT}" if on else f"1px solid {theme.BORDER}"  # the divider, or the accent
            # lighter than the panel's body: the panel reads as a window of its own
            self.setStyleSheet(f"#dockTitle {{ background: {theme.BG_ROLLOUT}; border-bottom: {line}; }}"
                               f"#dockTitle QLabel, #dockTitle QToolButton {{ background: transparent; }}")


class MainWindow(QMainWindow):
    def __init__(self, paths=(), channel: str | None = None):
        """`channel`: the name of the local channel to listen on (editor/remote.py), None for none."""
        super().__init__()
        from trafilatura3d.editor import wheel_guard
        wheel_guard.install()  # the wheel scrolls panels, it edits a value only with the focus
        self.setWindowTitle("Trafilatura 3D")
        self.resize(1600, 980)
        self.setAcceptDrops(True)  # files dropped on the window open as File > Open does (both workspaces)
        self.settings = QSettings()  # the application's organization and name (app.py; tests use their own)
        splash.step("Starting the GPU")
        self.doc = Document()
        self.doc.error.connect(self.status)
        self.doc.packages_changed.connect(self._packages_changed)
        self.doc.dirty_changed.connect(self._packages_changed)

        # Home screen and workspace (an inner main window holding the docks).
        self.stack = QStackedWidget()
        self.home = HomeScreen()
        self.home.new_graph.connect(self.new_graph_dialog)
        self.home.open_package.connect(self.open_package)
        self.home.new_project.connect(lambda: self.paint.new_dialog())
        self.home.open_project.connect(lambda: self.paint.open_dialog())
        self.home.open_recent.connect(self.open_path)
        self.home.close_home.connect(self._close_home)
        self.workspace = QMainWindow()
        self.workspace.setDockNestingEnabled(True)
        filler = QWidget()
        filler.setMaximumSize(0, 0)
        self.workspace.setCentralWidget(filler)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.workspace)
        self.setCentralWidget(self.stack)
        self.graph_shortcuts = []  # the Graph workspace's shortcut actions (off while Paint's is shown)

        splash.step("Graph workspace")
        self.graph = GraphPanel(self.doc, self.status)
        self.graph.export_outputs = self.export_outputs  # Graph view > Tools > Export Outputs…
        # Export outputs' Automatic Export: (id(package), graph id) -> (package, graph id, settings)
        self.auto_exports = {}
        self._auto_timer = QTimer(self, singleShot=True, interval=600, timeout=self._run_auto_exports)
        self.properties = PropertiesPanel(self.doc)
        splash.step("2D and 3D views")
        self.view2d = View2D(self.doc)
        self.view3d = View3D(self.doc)
        splash.step("Library")
        self.library = LibraryPanel(self.doc.ev)
        self.explorer = ExplorerPanel(self.doc, self)
        self.docks = {}
        for key, title, widget, ic in [("explorer", "Explorer", self.explorer, "explorer"),
                                       ("library", "Library", self.library, "library"),
                                       ("graph", "Graph", self.graph, "graph"),
                                       ("view3d", "3D View", self.view3d, "view3d"),
                                       ("view2d", "2D View", self.view2d, "view2d"),
                                       ("properties", "Properties", self.properties, "properties")]:
            dock = QDockWidget(title, self.workspace)
            dock.setObjectName(key)
            dock.setWidget(widget)
            dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
                             | QDockWidget.DockWidgetFloatable)
            dock.setTitleBarWidget(DockTitle(dock, ic, self))
            self.docks[key] = dock
        self._default_layout()
        self.default_state = self.workspace.saveState()
        state = self.settings.value(LAYOUT_KEY)
        if state is not None:
            self.workspace.restoreState(state)

        self.graph.title_changed.connect(lambda t: self._title("graph", t))
        self.properties.title_changed.connect(lambda t: self._title("properties", t))
        self.view2d.title_changed.connect(lambda t: self._title("view2d", t))
        self.view3d.title_changed.connect(lambda t: self._title("view3d", t))
        self.view3d.show_uvs.connect(lambda d: (self.view2d.set_uvs(d), self.docks["view2d"].show(),
                                                self.docks["view2d"].raise_()))
        self.explorer.open_scene.connect(self.show_scene)
        self.view3d.settings_requested.connect(lambda *a: (self.properties.show_settings(*a),
                                                           self.docks["properties"].show(),
                                                           self.docks["properties"].raise_()))
        self.graph.selection_changed.connect(self._selection)
        # What the views show after an edit: a step per turn of the event loop, at the preview resolution while a
        # value is dragged (Document.shown); the selection is what the next edit restarts from.
        self.doc.live = LiveCook(self, self.doc.ev, self._plan_graph, self._graph_step, paused=self._navigating,
                                 owner="graph selection")
        self.doc.changed.connect(self._graph_changed)
        self.graph.selection_changed.connect(lambda *_: self.doc.live.schedule())
        self.properties.open_function.connect(self.graph.open_function)
        self.properties.open_fxmap.connect(self.graph.open_fxmap)
        self.properties.open_function_target.connect(self.graph.open_function_target)
        self.properties.rename_graph.connect(self.rename_graph)
        self.graph.functions.selection_changed.connect(self._function_selection)
        self.graph.fxmaps.selection_changed.connect(self._fx_selection)
        self.graph.view_2d.connect(self.view2d.show_node)
        self.graph.view_3d.connect(self.view3d.show_graph)
        self.graph.view_3d_reset.connect(lambda p, g: self.view3d.show_graph(p, g, reset=True))
        self.graph.view_3d_node.connect(self.view3d.show_node)
        self.graph.graph_shown.connect(self._graph_shown)
        self.graph.node_created.connect(self._node_created)  # Preferences: 3D / 2D views follow the opened graph
        self.explorer.open_graph.connect(self.open_graph)
        self.explorer.graph_selected.connect(lambda p, g: self.properties.show_target(p, g, None))
        self.explorer.view_3d.connect(self.view3d.show_graph)
        self._menus()
        self._status_bar()
        splash.step("Paint workspace")
        from trafilatura3d.editor.paint.workspace import PaintWorkspace
        self.paint = PaintWorkspace(self, self.doc.ev)
        self.paint.title_changed.connect(self.setWindowTitle)
        self.stack.addWidget(self.paint)
        for a in self.paint.shortcut_actions:
            self.addAction(a)
        self._workspace_menus("graph")
        QShortcut(QKeySequence("Shift+Space"), self, activated=self._maximize_under_mouse)
        self.maximized = None

        self.backup_timer = QTimer(self, timeout=self._auto_backup)
        from trafilatura3d.editor import preferences
        self.view3d.view.look.show_env = not preferences.get("environment_hidden")  # 3D View preference
        from trafilatura3d.project import configuration  # Project settings > General: default normal format
        self.view3d.set_setting("opengl_normals", configuration().setting("general", "normal_format") == "OpenGL")
        for a in self.view3d.toggles.get("show_env", ()):
            a.setChecked(self.view3d.view.look.show_env)
        self.apply_preferences()
        from trafilatura3d.editor import plugin_host
        splash.step("Plugins")
        plugin_host.install(self)  # before the packages open: plugins hear package_opened
        for p in paths:
            splash.step(f"Opening {Path(p).name}")
            self.open_path(p)
        if self.paint.doc is None:
            self._show(self.workspace if self.doc.packages else self.home)
        self.remote = None
        if channel:
            from trafilatura3d.editor.remote import RemoteServer
            self.remote = RemoteServer(self, channel)

    # --- layout -------------------------------------------------------------------------------

    def _default_layout(self):
        w, d = self.workspace, self.docks
        # Columns first (Explorer | Graph | Properties), then split the left and middle columns.
        w.addDockWidget(Qt.LeftDockWidgetArea, d["explorer"])
        w.splitDockWidget(d["explorer"], d["graph"], Qt.Horizontal)
        w.splitDockWidget(d["graph"], d["properties"], Qt.Horizontal)
        w.splitDockWidget(d["explorer"], d["library"], Qt.Vertical)
        w.splitDockWidget(d["graph"], d["view3d"], Qt.Vertical)
        w.splitDockWidget(d["view3d"], d["view2d"], Qt.Horizontal)
        w.resizeDocks([d["explorer"], d["graph"], d["properties"]], [300, 900, 400], Qt.Horizontal)
        w.resizeDocks([d["explorer"], d["library"]], [380, 520], Qt.Vertical)
        w.resizeDocks([d["graph"], d["view3d"]], [520, 400], Qt.Vertical)
        for dock in d.values():
            dock.show()

    def reset_layout(self):
        self.maximized = None
        for dock in self.docks.values():
            dock.setFloating(False)
            dock.show()
        self.workspace.restoreState(self.default_state)

    def toggle_maximize(self, dock):
        if self.maximized is None:
            self.maximized = self.workspace.saveState()
            for other in self.docks.values():
                if other is not dock:
                    other.hide()
        else:
            self.workspace.restoreState(self.maximized)
            self.maximized = None

    def _maximize_under_mouse(self):
        w = QApplication.widgetAt(QCursor.pos())
        while w is not None and not isinstance(w, QDockWidget):
            w = w.parentWidget()
        if w is not None or self.maximized is not None:
            self.toggle_maximize(w)

    def _title(self, key, text):
        self.docks[key].titleBarWidget().set_title(text)

    # --- menus --------------------------------------------------------------------------------

    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu("File")
        new = f.addMenu("New")  # as the original: New graph (with its templates) or an empty package
        self._act(new, "New graph…", self.new_graph_dialog, QKeySequence.New)
        self._act(new, "Empty", self.new_package)
        self._act(f, "Open package…", self.open_package, QKeySequence.Open)
        self.recent_menu = f.addMenu("Recent packages")
        f.addSeparator()
        self._act(f, "Save", lambda: self._current_package() and self.save_package(self._current_package()),
                  QKeySequence.Save)
        self._act(f, "Save As…", lambda: self._current_package() and self.save_package(self._current_package(), ask=True),
                  QKeySequence.SaveAs)
        f.addSeparator()
        self._act(f, "Close package", lambda: self._current_package() and self.close_package(self._current_package()))
        self._act(f, "Home screen", self.show_home)
        f.addSeparator()
        self._act(f, "Quit", self.close, QKeySequence.Quit)
        e = mb.addMenu("Edit")
        undo = self.doc.undo.createUndoAction(self, "Undo")
        undo.setShortcuts(variants(QKeySequence.Undo))
        redo = self.doc.undo.createRedoAction(self, "Redo")
        redo.setShortcuts(variants(QKeySequence.Redo, "Ctrl+Shift+Z", "Ctrl+Y"))
        for a in (undo, redo):
            e.addAction(a)
            self.addAction(a)  # a menu is a window of its own: on the main window the shortcut works everywhere
        e.addSeparator()
        self._act(e, "Preferences…", self.open_preferences, QKeySequence.Preferences)
        t = mb.addMenu("Tools")
        self._act(t, "Clear the cooked results cache", self._clear_cache)
        self._act(t, "Plugin Manager…", self.open_plugin_manager)
        # Without it the warnings are found when a package opens, then only for the nodes being worked on
        from trafilatura3d.editor import preferences
        live = self._act(t, "Live warnings (slow)", lambda on: (preferences.put("live_warnings", on),
                                                               self.doc.set_live_warnings(on)))
        live.setCheckable(True)
        live.setChecked(preferences.get("live_warnings"))
        self.doc.live_warnings = live.isChecked()
        win = mb.addMenu("Windows")
        for dock in self.docks.values():
            win.addAction(dock.toggleViewAction())
        win.addSeparator()
        self._act(win, "Reset layout", self.reset_layout)
        self._act(win, "Unmaximise window", lambda: self.maximized is not None and self.toggle_maximize(None))
        h = mb.addMenu("Help")
        self._act(h, "About Trafilatura 3D", self.about)
        self.home_menus = self._home_menus()

        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QSize(16, 16))
        bar.addAction(icon("graph"), "New graph", self.new_graph_dialog)
        bar.addAction(icon("open"), "Open package", self.open_package)
        bar.addAction(icon("save"), "Save", lambda: self._current_package() and self.save_package(self._current_package()))
        bar.addSeparator()
        bar.addAction(icon("undo"), "Undo", self.doc.undo.undo)
        bar.addAction(icon("redo"), "Redo", self.doc.undo.redo)
        bar.addSeparator()
        bar.addAction(icon("home"), "Home screen", self.show_home)
        theme.divider(bar, "bottom")
        self.addToolBar(bar)
        self.graph_bar = bar
        self.graph_menus = [a.menu() for a in mb.actions() if a.menu() is not None]
        self.graph_shortcuts += [undo, redo]
        self._fill_recent()

    def _home_menus(self) -> list:
        """The Home screen's menus: generic, neither workspace's (what to open or create, preferences, help)."""
        f = QMenu("File", self)
        for text, slot in (("New graph…", self.new_graph_dialog), ("Open package…", self.open_package),
                           ("New paint project…", lambda: self.paint.new_dialog()),
                           ("Open paint project…", lambda: self.paint.open_dialog())):
            f.addAction(text, slot)
        f.addSeparator()
        f.addAction("Preferences…", self.open_preferences)
        f.addSeparator()
        f.addAction("Quit", self.close)
        h = QMenu("Help", self)
        h.addAction("About Trafilatura 3D", self.about)
        return [f, h]

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcuts(variants(shortcut))
            self.addAction(a)  # a menu is a window of its own: on the main window the shortcut works everywhere
            self.graph_shortcuts.append(a)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # --- workspaces -------------------------------------------------------------------------------

    def _menus_of(self, workspace: str) -> list:
        return {"graph": self.graph_menus, "paint": self.paint.menus}[workspace]

    def workspace_menu(self, workspace: str, title: str):
        """A workspace's ("graph" or "paint") top menu `title`, created at the end if missing (plugins add theirs
        here)."""
        menus = self._menus_of(workspace)
        found = next((m for m in menus if m.title().replace("&", "") == title), None)
        if found is None:
            found = QMenu(title, self)
            menus.append(found)
            self._workspace_menus(self.shown)
        return found

    def remove_workspace_menu(self, workspace: str, menu) -> None:
        self._menus_of(workspace).remove(menu)
        self._workspace_menus(self.shown)

    def _workspace_menus(self, shown: str):
        """The main menu, toolbar and shortcuts of what is shown ("home", "graph" or "paint"): each workspace
        has its own, the Home generic ones and no toolbar."""
        mb = self.menuBar()
        self.shown = shown
        mb.clear()
        for m in {"graph": self.graph_menus, "paint": self.paint.menus, "home": self.home_menus}[shown]:
            mb.addMenu(m)
        self.graph_bar.setVisible(shown == "graph")
        for a in self.graph_shortcuts:
            a.setEnabled(shown == "graph")
        for a in self.paint.shortcut_actions:
            a.setEnabled(shown == "paint")

    def _show(self, widget):
        """Show the Home, the Graph workspace or the Paint one, with its menus."""
        self.stack.setCurrentWidget(widget)
        self._workspace_menus("home" if widget is self.home else "graph" if widget is self.workspace else "paint")
        if widget is not self.paint:
            self._packages_changed()

    def show_home(self):
        self.home.refresh()
        self._show(self.home)

    def show_paint(self):
        self._show(self.paint)

    def paint_closed(self):
        """The Paint project was closed: the Graph workspace if packages are open, else the Home."""
        if self.stack.currentWidget() is self.paint:
            self._show(self.workspace) if self.doc.packages else self.show_home()

    def _close_home(self):
        if self.paint.doc is not None and not self.doc.packages:
            self._show(self.paint)
        else:
            self._show(self.workspace)

    def about(self):
        QMessageBox.about(self, "Trafilatura 3D", f"Trafilatura 3D {VERSION}\nProcedural texture graphs and texture painting on the "
                                           "GPU (wgpu).\n\nThumbnails of the Paint filters and generators: "
                                           "\"Shader-Ball\" by ZaNNoZa (https://sketchfab.com/3d-models/"
                                           "shader-ball-fc08e698fd5d4f66a10c15bb5c0359b0), CC BY 4.0 "
                                           "(https://creativecommons.org/licenses/by/4.0/), unchanged; subdivided "
                                           "and rendered by Trafilatura 3D.")

    def _status_bar(self):
        sb = self.statusBar()
        info = self.doc.ev.device.adapter.info
        self.engine = QLabel(f"Trafilatura 3D Engine: wgpu {info['backend_type']} ({info['device']})")
        self.memory = QLabel()
        sb.addPermanentWidget(self.engine)
        sb.addPermanentWidget(self.memory)
        sb.addPermanentWidget(QLabel(f"Version: {VERSION}"))
        self._mem_timer = QTimer(self, interval=1000, timeout=self._memory)
        self._mem_timer.start()

    def _memory(self):
        self.memory.setText(f"Memory: {self.doc.ev.cached_bytes() / 2**20:.0f} MB")

    def status(self, message: str):
        self.statusBar().showMessage(message, 8000)

    # --- packages -----------------------------------------------------------------------------

    def _current_package(self):
        return self.graph.package or (self.doc.packages[0] if self.doc.packages else None)

    def new_package(self):
        """File > New > Empty: an empty package in the Explorer (graphs come from New graph)."""
        self.doc.new_package()
        self._show(self.workspace)

    def new_graph_dialog(self, preselect=None):
        names = [self.doc.name(p) for p in self.doc.packages]
        dlg = NewGraphDialog(self.doc.packages, names, self, preselect if preselect in self.doc.packages else None)
        if not dlg.exec():
            return
        self.create_graph(*dlg.result_values())

    def create_graph(self, name: str, size, pkg=None, template: str = "Empty") -> tuple:
        """A new graph from a template (TEMPLATES) in `pkg`, or in a new package (None), shown in the Graph view (the
        New graph dialog, the remote channel); returns (package, graph id)."""
        if pkg is None:
            pkg, gid = self.doc.new_package(), name
            g = pkg.new_graph(gid, size[0])
            g.parent_size = list(size)
            shader = apply_template(g, template)
        else:
            gid = name
            n = 1
            while gid in pkg.graphs:
                n += 1
                gid = f"{name}_{n}"

            shader = None

            def do():
                nonlocal shader
                g = pkg.new_graph(gid, size[0])
                g.parent_size = list(size)
                shader = apply_template(g, template)
                return gid
            if self.doc.edit(pkg, "New graph", do) is None:
                raise DocumentError(f"cannot create graph {gid!r}")
        if shader is not None:  # the 3D View shows the template's material (before the graph: one cook)
            self.view3d.use_template(shader)
        self.open_graph(pkg, gid)
        return pkg, gid

    def rename_graph(self, pkg, old: str, new: str) -> bool:
        """Rename a graph (one undo step, its instances follow); the Graph view keeps showing it."""
        shown = self.graph.package is pkg and self.graph.graph_id == old
        if new == old or self.doc.edit(pkg, "Rename graph", lambda: pkg.rename_graph(old, new) or True) is None:
            return False
        if shown:
            self.open_graph(pkg, new)
        return True

    def open_package(self):
        from trafilatura3d.editor.preferences import suggested_dir
        name, _ = QFileDialog.getOpenFileName(self, "Open package", suggested_dir(),
                                              f"Packages (*{SUFFIX} *.sbs);;Trafilatura 3D package (*{SUFFIX});;"
                                              "Substance Designer package (*.sbs)")
        if name:
            self.open_path(name)

    @staticmethod
    def _openable(mime) -> list[str]:
        """The dropped files `open_path` opens: packages, .sbs files (imported), Paint projects, .spp projects
        (imported)."""
        from trafilatura3d.paint.model import PROJECT_SUFFIX
        return [p for p in (u.toLocalFile() for u in mime.urls())
                if p.lower().endswith((SUFFIX, ".sbs", PROJECT_SUFFIX, ".spp"))]

    def dragEnterEvent(self, e):  # what a panel doesn't take of a drag (its own drops) comes here
        if self._openable(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        self.dragEnterEvent(e)

    def dropEvent(self, e):
        paths = self._openable(e.mimeData())
        if not paths:
            e.ignore()
            return
        e.acceptProposedAction()
        for path in paths:
            self.open_path(path)

    def open_path(self, path):
        from trafilatura3d.paint.model import PROJECT_SUFFIX
        if str(path).endswith(PROJECT_SUFFIX) or str(path).lower().endswith(".spp"):  # the Paint workspace
            self.paint.open_path(path)
            return
        try:
            self.load_package(path)
        except (DocumentError, OSError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Trafilatura 3D", f"Cannot open {path}:\n{e}")

    def load_package(self, path):
        """Open a package (a .sbs is imported into a new unsaved one) and show its first graph; raises when it
        can't (DocumentError, OSError, ValueError, KeyError)."""
        designer = str(path).lower().endswith(".sbs")  # imported into a new unsaved package
        pkg = self.doc.import_sbs(path) if designer else self.doc.open(path)
        if designer:
            self.status(f"Imported {Path(path).name}: {len(pkg.notes)} import notes, see the graphs' warnings")
        else:
            self.remember(pkg.path)
        if pkg.graphs:
            self.open_graph(pkg, next(iter(pkg.graphs)))
        return pkg

    # --- Export outputs ----------------------------------------------------------------------------

    def export_outputs(self, pkg, gid=None):
        """The Export outputs dialog for a graph (the one in the Graph view when not given)."""
        from trafilatura3d.editor.export_dialog import ExportDialog
        if gid is None:
            pkg, gid = self.graph.package, self.graph.graph_id
        if pkg is None or gid not in pkg.graphs:
            self.status("Export outputs: open a graph first")
            return
        dlg = ExportDialog(self.doc, pkg, gid, self, on_automatic=self._set_auto_export)
        if dlg.exec() and getattr(dlg, "written", None) is not None:
            self.status(f"Exported {len(dlg.written)} files to {dlg.settings()['destination']}")

    def _set_auto_export(self, pkg, gid, settings):
        key = (id(pkg), gid)
        if settings is None:
            self.auto_exports.pop(key, None)
        else:
            self.auto_exports[key] = (pkg, gid, settings)
        if self.auto_exports and not getattr(self, "_auto_connected", False):
            self.doc.changed.connect(self._auto_changed)
            self._auto_connected = True

    def _auto_changed(self, pkg):
        if any(p is pkg for p, _, _ in self.auto_exports.values()):
            self._auto_timer.start()

    def _run_auto_exports(self):
        from trafilatura3d.editor.export_dialog import run_export
        for pkg, gid, settings in list(self.auto_exports.values()):
            if pkg not in self.doc.packages or gid not in pkg.graphs:
                continue
            try:
                written = run_export(self.doc, pkg, gid, settings)
                self.status(f"Automatic Export: {gid}, {len(written)} files")
            except Exception as e:  # noqa: BLE001 - shown in the status bar, the next change tries again
                self.status(f"Automatic Export: {gid}: {e}")

    def open_graph(self, pkg, gid):
        from trafilatura3d.editor.export_dialog import load_settings
        if pkg.path is not None and load_settings(pkg, gid)["automatic"]:  # Automatic Export kept on for it
            self._set_auto_export(pkg, gid, load_settings(pkg, gid))
        self._show(self.workspace)
        self.graph.open_graph(pkg, gid)
        self.properties.show_target(pkg, gid, None)

    def save_package(self, pkg, ask=False, path=None):
        path = path or pkg.path
        if ask or path is None:
            from trafilatura3d.editor.preferences import suggested_dir
            start = str(path) if path else str(Path(suggested_dir()) / f"untitled{SUFFIX}")
            name, _ = QFileDialog.getSaveFileName(self, "Save package", start,
                                                  f"Trafilatura 3D package (*{SUFFIX})")
            if not name:
                return False
            path = Path(name).with_suffix(SUFFIX)
        try:
            self.doc.save(pkg, path)
        except (OSError, ValueError) as e:  # ValueError: a plugin's Veto (its validation refused the save)
            QMessageBox.critical(self, "Trafilatura 3D", str(e))
            return False
        self.remember(pkg.path)
        self.status(f"Saved {pkg.path}")
        return True

    def close_package(self, pkg):
        if id(pkg) in self.doc.dirty:
            answer = QMessageBox.question(self, "Trafilatura 3D", f"Save the changes to {self.doc.name(pkg)}?",
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self.save_package(pkg)):
                return False
        self.graph.close_package(pkg)
        self.doc.close(pkg)
        if not self.doc.packages:
            self.show_paint() if self.paint.doc is not None else self.show_home()
        return True

    def show_scene(self, pkg, rid):
        """A package's 3D scene resource in the 3D View, its dock shown (the Explorer's double click, plugins)."""
        self.view3d.load_resource(pkg, rid)
        self.docks["view3d"].show()
        self.docks["view3d"].raise_()

    def _graph_shown(self, pkg, gid):
        from trafilatura3d.editor import preferences
        if preferences.get("outputs_3d_on_open"):
            self.view3d.show_graph(pkg, gid)
        if preferences.get("output_2d_on_open") and gid in pkg.graphs:
            outs = pkg.graphs[gid].output_nodes()
            if outs:
                self.view2d.show_node(pkg, gid, outs[0].id)

    def _node_created(self, pkg, gid, nid):
        from trafilatura3d.editor import preferences
        if preferences.get("display_new_node"):  # Automatically display newly created compositing node
            self.view2d.show_node(pkg, gid, nid)

    def open_preferences(self):
        from trafilatura3d.editor.preferences import PreferencesDialog
        PreferencesDialog(self).exec()

    def open_plugin_manager(self):
        from trafilatura3d.editor.plugin_host import PluginManager
        PluginManager(self).exec()

    def apply_preferences(self):
        """Apply the Preferences and the project settings to the running editor."""
        from trafilatura3d import compiler, project
        from trafilatura3d.editor import preferences as prefs
        if self.doc.undo.count() == 0:
            self.doc.undo.setUndoLimit(prefs.get("history_size"))
        self.doc.ev.budget = prefs.cache_bytes()
        compiler.SIZE_LIMIT_LOG2 = max(0, prefs.get("cooking_size_limit").bit_length() - 1)
        minutes = prefs.get("auto_backup_minutes")
        if minutes > 0:
            self.backup_timer.start(minutes * 60_000)
        else:
            self.backup_timer.stop()
        self.graph.apply_preferences()
        project.set_configuration(project.Configuration(project.configuration().config_path))
        self.view2d.fill_color_spaces()
        self.view3d.fill_color_spaces()
        self.library._fill_tree()
        self.library.refresh()
        self.doc._compiled.clear()
        for pkg in self.doc.packages:
            self.doc.changed.emit(pkg)

    def _auto_backup(self):
        """Auto Backup: unsaved packages copied to an .autosave folder next to them, keeping
        the last N versions (`backups`: the Paint's autosaves follow the same convention)."""
        from trafilatura3d.editor import backups
        from trafilatura3d.editor import preferences as prefs
        keep = prefs.get("auto_backup_versions")
        for pkg in self.doc.packages:
            if id(pkg) not in self.doc.dirty or pkg.path is None:
                continue
            try:
                target = backups.backup_path(pkg.path.parent / backups.FOLDER, pkg.path.stem, SUFFIX, keep)
                target.write_text(json.dumps(pkg.to_dict(), indent=2) + "\n", encoding="utf-8")
            except OSError as e:
                self.status(f"Auto-backup: {e}")

    def remember(self, path):
        if path is None:
            return
        recent = [str(path)] + [p for p in self.settings.value("recent", [], type=list) if p != str(path)]
        from trafilatura3d.editor import preferences
        self.settings.setValue("recent", recent[:preferences.get("recent_count")])
        self.settings.setValue("last_dir", str(Path(path).parent))
        self._fill_recent()
        self.home.refresh()

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in self.settings.value("recent", [], type=list):
            self.recent_menu.addAction(p, lambda p=p: self.open_path(p))

    def _packages_changed(self):
        if getattr(self, "paint", None) is not None and self.stack.currentWidget() is self.paint:
            return  # the Paint workspace titles the window
        dirty = any(id(p) in self.doc.dirty for p in self.doc.packages)
        self.setWindowTitle(("Trafilatura 3D Graph" if self.doc.packages else "Trafilatura 3D") + (" *" if dirty else ""))

    def _selection(self, pkg, gid, node_id):
        self.properties.show_target(pkg, gid, node_id)  # the 2D view follows double clicks

    # --- the Graph workspace's refresh (`LiveCook`) ----------------------------------------------

    THUMBNAIL_STEP = 16  # node thumbnails computed per step of a refresh

    def _navigating(self) -> bool:
        """A view being dragged (the 3D or 2D view's camera, the graph's pan or zoom): the refresh waits, the GPU and
        the event loop left to the frames (a 3D frame queued behind a cooking job waited up to 200 ms)."""
        return (self.view3d.view._drag is not None or self.view2d.view._drag is not None
                or self.graph.view._pan is not None or self.graph.view._zoom_drag is not None)

    def _graph_changed(self, pkg):
        if pkg not in (self.graph.package, self.view2d.package, self.view3d.package):
            return
        if self.doc.dragging:
            self.doc.live.previewed()
        else:
            self.doc.live.changed()

    def _plan_graph(self) -> Plan:
        """After an edit: the 2D view's node, the 3D view's channels a step each, then (at full resolution, when the
        Preferences compute them) the node thumbnails THUMBNAIL_STEP at a time."""
        from trafilatura3d.editor import preferences as prefs
        steps, related = [], []
        job, _ = self.view2d.source()
        if job is not None:
            steps.append(("2d", [job]))
            related.append(job)
        channels = list({job.key: job for job, _ in self.view3d.sources()}.values())
        steps += [("cook", [job]) for job in channels]
        steps.append(("3d", None))
        related += channels
        if not self.doc.live.previewing and prefs.get("auto_thumbnails"):
            stale = self.graph.stale_thumbnails()
            n = self.THUMBNAIL_STEP
            steps += [("thumbnails", stale[i:i + n]) for i in range(0, len(stale), n)]
            related += [job for _, job, _ in stale]
        return Plan(steps, related, self._graph_below())

    def _graph_below(self) -> list:
        """What an edit of the selected nodes restarts from: their inputs, at full resolution and as shown."""
        g = self.graph
        if g.graph is None:
            return []
        ids = g.scene.selected_ids()
        below = {}
        for compiled in (self.doc.compiled(g.package, g.graph_id), self.doc.shown(g.package, g.graph_id)):
            for node_id in ids if compiled is not None else ():
                for job, _ in compiled.nodes.get(node_id, {}).values():
                    below.update((src[0].key, src[0]) for src in job.inputs.values() if src)
        return list(below.values())

    def _graph_step(self, item):
        """A step of the refresh, a generator: a turn of the event loop at each wait of its cook for the GPU."""
        kind, data = item
        live = self.doc.live
        try:
            if kind == "cook":  # a channel of the 3D view: shown as soon as it is cooked
                yield from live.cook_steps(data)
                self.view3d.show_ready(data[0])
            elif kind == "2d":
                yield from live.cook_steps(data)
                self.view2d.refresh()
            elif kind == "3d":  # every channel cooked together (one by one, a cook could give back the others)
                yield from live.cook_steps(job for job, _ in self.view3d.sources())
                if not self.view3d.shows_all():  # each channel already shown as it came (`show_ready`)
                    self.view3d.refresh()
            else:
                yield from self.graph.thumbnail_steps(data, live.related, live.precious)
        except Exception as e:  # noqa: BLE001 - a failing node must not stop the editor
            self.status(f"Cook error: {e}")

    def _function_selection(self, target):
        if target is None:
            fp = self.graph.functions
            self.properties.show_target(fp.package, fp.graph_id, fp.node_id)
        else:
            self.properties.show_function_node(*target)

    def _fx_selection(self, target):
        if target is None:
            fx = self.graph.fxmaps
            self.properties.show_target(fx.package, fx.graph_id, fx.node_id)
        else:
            self.properties.show_fx_node(*target)

    def _clear_cache(self):
        self.doc.ev.clear()
        self.status("Cooked results cache cleared")

    def closeEvent(self, e):
        if not self.paint.close_project():
            e.ignore()
            return
        self.paint.save_layout()
        for pkg in list(self.doc.packages):
            if not self.close_package(pkg):
                e.ignore()
                return
        self.settings.setValue(LAYOUT_KEY, self.workspace.saveState() if self.maximized is None else self.maximized)
        from trafilatura3d import plugins
        plugins.manager().unload_all()  # their uninitialize_plugin runs
        if self.remote is not None:
            self.remote.close()
        e.accept()
