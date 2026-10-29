"""The Explorer: the open packages and their graphs. Double click opens a graph in the
Graph view; graphs can be dragged into a graph to create instances; context menus create, rename,
duplicate, remove and export graphs and save or close packages."""

from __future__ import annotations

import copy

from PySide6.QtCore import QByteArray, QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QDrag, QFont
from PySide6.QtWidgets import (QFileDialog, QInputDialog, QMenu, QMessageBox, QToolBar, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.editor.icons import icon
from trafilatura3d.package import Graph, INSTANCE


class _Tree(QTreeWidget):
    def startDrag(self, actions):
        item = self.currentItem()
        data = item.data(0, Qt.UserRole) if item else None
        if not data or data[0] not in ("graph", "resource"):
            return
        mime = QMimeData()
        # A graph becomes an instance, a bitmap resource a Bitmap node.
        kind = "application/x-trafilatura3d-graph" if data[0] == "graph" else "application/x-trafilatura3d-resource"
        mime.setData(kind, QByteArray(f"{data[1]}\n{data[2]}".encode()))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class ExplorerPanel(QWidget):
    open_graph = Signal(object, str)  # package, graph id
    graph_selected = Signal(object, str)
    view_3d = Signal(object, str)
    open_scene = Signal(object, str)  # package, 3D scene resource id

    def __init__(self, document, window):
        super().__init__()
        self.doc, self.window = document, window
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = theme.divider(QToolBar(), "bottom")
        bar.setIconSize(QSize(16, 16))
        bar.addAction(icon("save"), "Save", self.save_selected)
        lay.addWidget(bar)
        self.tree = _Tree()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemDoubleClicked.connect(self._double)
        self.tree.currentItemChanged.connect(self._current)
        lay.addWidget(self.tree)
        self.doc.packages_changed.connect(self.refresh)
        # graphs added, removed or renamed (not while a value is dragged: its release refreshes)
        self.doc.changed.connect(lambda _: None if self.doc.dragging else self.refresh())
        self.doc.warnings_changed.connect(lambda _: self.refresh())
        self.doc.dirty_changed.connect(self.names)

    def names(self):
        """Only the packages' names, with their unsaved mark: an arrangement or undo step changes nothing else."""
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            top.setText(0, self.doc.name(self.doc.packages[top.data(0, Qt.UserRole)[1]]))

    def refresh(self):
        expanded = {self.tree.topLevelItem(i).text(0).rstrip("*") for i in range(self.tree.topLevelItemCount())
                    if self.tree.topLevelItem(i).isExpanded()}
        self.tree.clear()
        for index, pkg in enumerate(self.doc.packages):
            # as last found (`Document.check`, `check_nodes`): never computed here, at each edit
            warns = {gid: w for gid in ("", *pkg.graphs) if (w := self.doc.graph_warnings(pkg, gid))}
            top = QTreeWidgetItem([self.doc.name(pkg)])
            top.setIcon(0, icon("package"))
            f = QFont()
            f.setItalic(True)
            f.setBold(True)
            top.setFont(0, f)
            top.setData(0, Qt.UserRole, ("package", index))
            top.setToolTip(0, str(pkg.path) if pkg.path else "Not saved yet")
            if warns:  # a warning icon whose tooltip lists the warnings
                top.setIcon(0, icon("warning"))
                top.setToolTip(0, top.toolTip(0) + "".join(
                    f"\n⚠ {w.message}" + (f" ({w.detail})" if w.detail else "") for w in warns.get("", []))
                               + ("\n⚠ some graphs have warnings" if set(warns) - {""} else ""))
            self.tree.addTopLevelItem(top)
            for gid, g in pkg.graphs.items():
                it = QTreeWidgetItem([g.label or gid])
                it.setIcon(0, icon("warning" if gid in warns else "graph"))
                if gid in warns:
                    it.setToolTip(0, "\n".join(f"⚠ {w.message}" + (f" [{w.node}]" if w.node else "")
                                               + (f" ({w.detail})" if w.detail else "") for w in warns[gid]))
                it.setData(0, Qt.UserRole, ("graph", index, gid))
                top.addChild(it)
            for rid, r in pkg.resources.items():
                it = QTreeWidgetItem([rid])
                it.setIcon(0, icon("view3d" if r.kind == "scene" else "bitmap"))
                it.setToolTip(0, r.path)
                if not r.path.startswith("primitive:") and not _resource_exists(pkg, r.path):
                    it.setIcon(0, icon("warning"))
                    it.setToolTip(0, f"⚠ Linked file not found\n{r.path}")
                it.setData(0, Qt.UserRole, ("resource", index, rid))
                top.addChild(it)
            top.setExpanded(not expanded or self.doc.name(pkg).rstrip("*") in expanded)

    def _target(self, item=None):
        item = item or self.tree.currentItem()
        if item is None:
            return None, None
        data = item.data(0, Qt.UserRole)
        pkg = self.doc.packages[data[1]]
        return pkg, (data[2] if data[0] == "graph" else None)

    def _double(self, item, _col):
        pkg, gid = self._target(item)
        data = item.data(0, Qt.UserRole)
        if gid:
            self.open_graph.emit(pkg, gid)
        elif data[0] == "resource" and pkg.resources[data[2]].kind == "scene":
            self.open_scene.emit(pkg, data[2])

    def _current(self, item, _prev):
        if item is None:
            return
        pkg, gid = self._target(item)
        if gid:
            self.graph_selected.emit(pkg, gid)

    # --- actions ------------------------------------------------------------------------------

    def _menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QMenu(self)
        if item is None:
            menu.addAction("New Package", self.window.new_package)
            menu.addAction("Open Package…", self.window.open_package)
        else:
            pkg, gid = self._target(item)
            if gid is None and item.data(0, Qt.UserRole)[0] == "package":
                new = menu.addMenu("New")
                new.addAction("Graph…", lambda: self.new_graph(pkg))
                menu.addAction("Link bitmap…", lambda: self.link_bitmap(pkg))
                menu.addAction("Link 3D scene…", lambda: self.link_scene(pkg))
                menu.addSeparator()
                menu.addAction("Dependency manager", lambda: self._dependency_manager(pkg))
                menu.addAction("Reload", lambda: self._reload(pkg))
                menu.addSeparator()
                menu.addAction("Save", lambda: self.window.save_package(pkg))
                menu.addAction("Save As…", lambda: self.window.save_package(pkg, ask=True))
                menu.addSeparator()
                menu.addAction("Close", lambda: self.window.close_package(pkg))
            elif item.data(0, Qt.UserRole)[0] == "resource":
                rid = item.data(0, Qt.UserRole)[2]
                menu.addAction("Relocate…", lambda: self._relocate_resource(pkg, rid))
                if pkg.resources[rid].kind == "scene":
                    menu.addAction("Open in 3D View", lambda: self.open_scene.emit(pkg, rid))
                    bake = menu.addAction("Bake model information", lambda: self.bake_model(pkg, rid))
                    if pkg.resources[rid].path.startswith("primitive:"):
                        bake.setEnabled(False)
                        bake.setToolTip("Bakers read a mesh file: link one with Link 3D scene…")
                menu.addSeparator()
                menu.addAction("Remove", lambda: self.remove_resource(pkg, rid))
            elif gid is not None:
                menu.addAction("Open", lambda: self.open_graph.emit(pkg, gid))
                menu.addAction("Rename…", lambda: self.rename_graph(pkg, gid))
                menu.addAction("Duplicate", lambda: self.duplicate_graph(pkg, gid))
                menu.addAction("Remove", lambda: self.remove_graph(pkg, gid))
                menu.addSeparator()
                menu.addAction("View Outputs in 3D View", lambda: self.view_3d.emit(pkg, gid))
                menu.addAction("Export Outputs as bitmaps", lambda: self.window.export_outputs(pkg, gid))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _dependency_manager(self, pkg):
        from trafilatura3d.editor.dependencies import DependencyManager
        DependencyManager(self.doc, pkg, self).exec()

    def _reload(self, pkg):
        from trafilatura3d.editor.dependencies import reload_package
        reload_package(self.doc, pkg, self)

    def _relocate_resource(self, pkg, rid):
        from trafilatura3d.diagnostics import relocate
        r = pkg.resources[rid]
        from trafilatura3d.meshio import MESH_FILTER
        filt = MESH_FILTER if r.kind == "scene" else \
            "Images (*.png *.jpg *.jpeg *.tga *.tif *.tiff *.exr *.bmp *.psd *.hdr)"
        path, _ = QFileDialog.getOpenFileName(self, f"Relocate {rid}", "", filt)
        if path:
            self.doc.edit(pkg, "Relocate", lambda: relocate(pkg, r.path, path))

    def new_graph(self, pkg):
        self.window.new_graph_dialog(pkg)

    def rename_graph(self, pkg, gid):
        new, ok = QInputDialog.getText(self, "Rename graph", "Identifier", text=gid)
        if not ok or new == gid:
            return

        self.window.rename_graph(pkg, gid, new)

    def duplicate_graph(self, pkg, gid):
        def do():
            n, new = 1, f"{gid}_copy"
            while new in pkg.graphs:
                n += 1
                new = f"{gid}_copy{n}"
            g = Graph(new, pkg)
            g.load_dict(copy.deepcopy(pkg.graphs[gid].to_dict()))
            pkg.graphs[new] = g
        self.doc.edit(pkg, "Duplicate graph", do)

    def remove_graph(self, pkg, gid):
        users = [g.id for g in pkg.graphs.values() for n in g.nodes.values() if n.type == INSTANCE and n.ref == gid]
        if users:
            QMessageBox.warning(self, "Remove graph", f"{gid} is used by: {', '.join(sorted(set(users)))}")
            return
        self.doc.edit(pkg, "Remove graph", lambda: pkg.remove_graph(gid))

    def remove_resource(self, pkg, rid):
        """A linked resource out of the package (one undo step); refused while nodes name its file. Its file stays on
        disk."""
        users = sorted({f"{ref.user[0]}.{ref.user[1]}" for ref in pkg.file_references()
                        if ref.user and ref.resource == rid})
        if users:
            QMessageBox.warning(self, "Remove resource", f"{rid} is used by: {', '.join(users)}")
            return
        view3d = self.window.view3d
        if view3d.bake_resource == (pkg, rid):  # the scene on view: no longer a resource (its bakes, its settings)
            view3d.bake_resource = view3d.baked_preview = None
        self.doc.edit(pkg, "Remove resource", lambda: pkg.resources.pop(rid))

    def link_bitmap(self, pkg):
        if pkg.path is None:
            QMessageBox.information(self, "Link bitmap", "Save the package first: resources are relative to it.")
            return
        name, _ = QFileDialog.getOpenFileName(self, "Link bitmap", str(pkg.path.parent),
                                              "Images (*.png *.jpg *.jpeg *.tga *.tif *.tiff *.exr *.bmp *.psd *.hdr)")
        if not name:
            return
        import os
        from pathlib import Path
        from trafilatura3d.package import Resource
        rel = os.path.relpath(name, pkg.path.parent).replace(os.sep, "/")
        rid = Path(name).stem
        self.doc.edit(pkg, "Link bitmap", lambda: pkg.resources.__setitem__(rid, Resource("bitmap", rel)))

    def link_scene(self, pkg):
        """A 3D scene resource: a mesh file the 3D View opens on double click."""
        if pkg.path is None:
            QMessageBox.information(self, "Link 3D scene", "Save the package first: resources are relative to it.")
            return
        from trafilatura3d.meshio import MESH_FILTER
        name, _ = QFileDialog.getOpenFileName(self, "Link 3D scene", str(pkg.path.parent), MESH_FILTER)
        if not name:
            return
        import os
        from pathlib import Path
        from trafilatura3d.package import Resource
        rel = os.path.relpath(name, pkg.path.parent).replace(os.sep, "/")
        rid = Path(name).stem
        self.doc.edit(pkg, "Link 3D scene", lambda: pkg.resources.__setitem__(rid, Resource("scene", rel)))

    def bake_model(self, pkg, rid):
        """The Baking window, with the mesh in the 3D View: its cage and baked maps follow the settings."""
        from trafilatura3d.editor.bakers import open_baker
        if self.window.view3d.bake_resource != (pkg, rid):
            self.open_scene.emit(pkg, rid)
        open_baker(self.doc, pkg, rid, self)

    def save_selected(self):
        pkg, _ = self._target()
        if pkg is not None:
            self.window.save_package(pkg)



def _resource_exists(pkg, path):
    from pathlib import Path
    p = Path(path)
    if not p.is_absolute() and pkg.path is not None:
        p = pkg.path.parent / p
    return p.exists()
