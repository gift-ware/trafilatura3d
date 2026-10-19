"""The Graph panel: tabs of open graphs, two toolbar rows (graph tools; node palette and
Parent Size), the view, and the controller that turns gestures into document edits. Thumbnails are
recomputed (on the GPU, read back small) shortly after every change."""

from __future__ import annotations

import html
import json

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMenu, QStackedWidget,
                               QTabBar, QToolBar, QToolButton, QVBoxLayout, QWidget)

from trafilatura3d import graphtools
from trafilatura3d.editor import theme
from trafilatura3d.editor.graph.function_panel import FunctionPanel, FxFunction, NodeFunction
from trafilatura3d.editor.graph.fxmap_panel import FxMapPanel
from trafilatura3d.editor.graph.items import NodeItem, title_color
from trafilatura3d.editor.graph.nodemenu import NodeMenu, library_entries
from trafilatura3d.editor.library import library_icon
from trafilatura3d.editor.graph.scene import GraphScene
from trafilatura3d.editor.graph.view import GraphView
from trafilatura3d.editor.icons import icon
from trafilatura3d.editor.keys import shortcut
from trafilatura3d.evaluator import run
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.base import PortType
from trafilatura3d.package import INPUT_TYPES, INSTANCE, DocumentError

# The node palette: atomic nodes with their three-letter tags.
PALETTE = [("Bmp", "bitmap"), ("Bld", "blend"), ("Blr", "blur"), ("ChS", "channel_shuffle"), ("Cur", "curve"),
           ("DBl", "directional_blur"), ("DWr", "directional_warp"), ("Dst", "distance"), ("Emb", "emboss"),
           ("StM", "stamp_map"), ("GrD", "gradient_dynamic"), ("Gra", "gradient_map"), ("Gry", "grayscale_conversion"),
           ("HSL", "hsl"), ("Lvl", "levels"), ("Nrm", "normal"), ("Pix", "pixel_processor"), ("Shp", "sharpen"),
           ("SVG", "svg"), ("Txt", "text"), ("Trs", "transformation_2d"), ("Clr", "uniform_color"),
           ("Val", "value_processor"), ("Wrp", "warp"), ("InC", "input_color"), ("InG", "input_grayscale"),
           ("Out", "output")]
SIZES = [str(1 << k) for k in range(4, 14)]
DEPTH_TAG = {"8": "8", "16": "16", "16f": "16F", "32f": "32F"}


def tag_icon(text: str, color: QColor) -> QIcon:
    pm = QPixmap(48, 40)
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(color)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(1, 1, 22, 18, 2, 2)
    f = QFont()
    f.setPointSizeF(7.5)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor("#f4f4f4"))
    p.drawText(1, 1, 22, 18, Qt.AlignCenter, text)
    p.end()
    return QIcon(pm)


class GraphPanel(QWidget):
    selection_changed = Signal(object, str, object)  # package, graph id, node id or None
    view_2d = Signal(object, str, str)  # package, graph id, node id
    view_3d = Signal(object, str)  # package, graph id
    node_created = Signal(object, str, str)  # package, graph id, node id (Preferences: display it in 2D)
    export_outputs = None  # callable(package, graph id): the Export outputs dialog (set by the main window)
    view_3d_reset = Signal(object, str)  # package, graph id: back to the graph's outputs
    view_3d_node = Signal(object, str, str, object)  # package, graph id, node id, material slot (None: preview)
    graph_shown = Signal(object, str)  # the graph now shown (the 3D view follows it)
    title_changed = Signal(str)

    def __init__(self, document, status):
        super().__init__()
        self.doc = document
        self.status = status  # callable(message)
        self.tabs_data: list[tuple[object, str]] = []
        self.package = None
        self.graph_id = ""
        self.highlight_flow = False
        self.clipboard = None
        self.scene = GraphScene(self)
        self.view = GraphView(self.scene)
        self.warnings_label = QLabel(self.view)
        self.warnings_label.setStyleSheet("color: #e8a33a; background: rgba(20,20,20,200); padding: 4px 6px;"
                                          "border-radius: 3px;")
        self.warnings_label.setTextFormat(Qt.RichText)
        self.warnings_label.linkActivated.connect(self._warning_link)  # a node's warning frames that node
        self.warnings_label.hide()
        self.dismissed_warnings = {}  # (package id, graph id) -> warnings closed with the ✕, hidden while the same
        self.view.resized = self._place_warnings

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.tabbar = QTabBar()
        self.tabbar.setDrawBase(False)  # the divider under the tabs, not the style's base line
        self.tabbar.setTabsClosable(True)
        self.tabbar.setExpanding(False)
        self.tabbar.currentChanged.connect(self._tab_changed)
        self.tabbar.tabCloseRequested.connect(self.close_tab)
        tabs = theme.divider(QWidget(), "bottom")  # a tab bar paints itself: the line is on its holder
        tabs_lay = QVBoxLayout(tabs)
        tabs_lay.setContentsMargins(0, 0, 0, 0)
        tabs_lay.addWidget(self.tabbar)
        lay.addWidget(tabs)
        # Two pages: the graph (toolbars + view) and the function graph editor.
        self.pages = QStackedWidget()
        graph_page = QWidget()
        gl = QVBoxLayout(graph_page)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(0)
        gl.addWidget(theme.divider(self._tools_row(), "bottom"))
        self.palette_row = theme.Overflow(theme.divider(self._palette_row(), "bottom"))
        gl.addWidget(self.palette_row)
        gl.addWidget(self.view, 1)
        self.pages.addWidget(graph_page)
        self.functions = FunctionPanel(document, status)
        self.functions.back.connect(self.close_function)
        self.pages.addWidget(self.functions)
        self.fxmaps = FxMapPanel(document, status)
        self.fxmaps.back.connect(lambda: self.pages.setCurrentIndex(0) or self.sync())
        self.pages.addWidget(self.fxmaps)
        self._function_from_fx = False
        lay.addWidget(self.pages, 1)

        self.view.setAcceptDrops(True)
        self.view.on_drop = self._drop
        # a graph shown: its thumbnails as steps of the refresh (`LiveCook`), the interface answering meanwhile
        self.thumb_timer = QTimer(self, singleShot=True, interval=40, timeout=self._schedule_thumbnails)
        self.scene.selectionChanged.connect(self._selection)
        self.doc.changed.connect(self._doc_changed)
        self.doc.layout_changed.connect(self._layout_changed)
        self._arranging = False  # this panel's own layout edit: the scene is already arranged
        # The graph's shortcuts act on the view with the focus (it takes it when the mouse enters, see
        # GraphView.enterEvent): a text field elsewhere keeps its own Ctrl+C / Tab, and the function and
        # FX-Map editors' views have theirs.
        local = Qt.WidgetWithChildrenShortcut
        menu = lambda: self.open_node_menu(self.view.cursor_scene_pos)
        shortcut("Space", self.view, menu, local)
        self.tab_shortcut = QShortcut(QKeySequence("Tab"), self.view, activated=menu, context=local)
        self.node_shortcuts = []  # Preferences > Shortcuts: keys that create a node at the mouse
        self.drag_connector_menu = True
        shortcut(QKeySequence.Copy, self.view, self.copy, local)
        shortcut(QKeySequence.Paste, self.view, lambda: self.paste(self.view.cursor_scene_pos), local)
        shortcut("Ctrl+D", self.view, self.duplicate, local)
        shortcut("Ctrl+F", self.view, self.find_node, local)
        shortcut("Ctrl+E", self.view, self.edit_inner_graph, local)
        shortcut("Shift+D", self.view, self.toggle_disabled, local)
        shortcut("F2", self.view, lambda: self.go_to_pin(1), local)
        shortcut("Shift+F2", self.view, lambda: self.go_to_pin(-1), local)

    # --- toolbars -----------------------------------------------------------------------------

    def _tools_row(self) -> QToolBar:
        bar = QToolBar()
        bar.setIconSize(QSize(16, 16))
        bar.addAction(icon("focus"), "Focus selection (F)", self.view.focus)
        bar.addAction(icon("reset_zoom"), "Reset zoom (Z)", self.view.reset_zoom)
        bar.addAction(icon("camera"), "Export Graph view", self.export_view)
        tools = QToolButton()  # the Tools button
        tools.setIcon(icon("tools"))
        tools.setToolTip("Tools")
        tools.setPopupMode(QToolButton.InstantPopup)
        tmenu = QMenu(tools)
        tmenu.addAction("Export Outputs…", lambda: self.export_outputs and self.export_outputs(self.package, self.graph_id))
        tools.setMenu(tmenu)
        bar.addWidget(tools)
        info = QToolButton()
        info.setIcon(icon("info"))
        info.setToolTip("Node information")
        info.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(info)
        # The Node information menu
        self.act_badges = menu.addAction("Display node badges")
        self.act_names = menu.addAction("Display connector name")
        self.act_size = menu.addAction("Display node size")
        self.act_timings = menu.addAction("Display timings")
        self.act_limit = menu.addAction("Limit text scaling when zooming out")
        for a in (self.act_badges, self.act_names, self.act_size, self.act_limit):
            a.setCheckable(True)
            a.setChecked(True)
            a.toggled.connect(lambda _on: self._info_changed())
        self.act_timings.setCheckable(True)
        self.act_timings.toggled.connect(self._timings_toggled)
        menu.addAction("Reset timings", self.reset_timings)
        info.setMenu(menu)
        bar.addWidget(info)
        bar.addAction(icon("search"), "Node finder (Ctrl+F)", self.find_node)
        flow = bar.addAction(icon("highlight"), "Highlight flow")
        flow.setCheckable(True)
        flow.toggled.connect(self._flow)
        pal = bar.addAction(icon("palette"), "Node palette")
        pal.setCheckable(True)
        pal.setChecked(True)
        pal.toggled.connect(lambda on: self.palette_row.setVisible(on))  # the row is made below
        rect = bar.addAction(icon("links"), "Rectangle links")
        rect.setCheckable(True)
        rect.toggled.connect(self._rect_links)
        align = QToolButton()
        align.setIcon(icon("align"))
        align.setToolTip("Node alignment tools")
        align.setPopupMode(QToolButton.InstantPopup)
        amenu = QMenu(align)
        for label, mode in [("Align left", "left"), ("Align horizontal centers", "hcenter"), ("Align right", "right"),
                            ("Align top", "top"), ("Align vertical centers", "vcenter"), ("Align bottom", "bottom"),
                            ("Distribute horizontally", "hdist"), ("Distribute vertically", "vdist")]:
            amenu.addAction(label, lambda m=mode: self.align(m))
        align.setMenu(amenu)
        bar.addWidget(align)
        return bar

    def _palette_row(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 0, 4, 2)
        lay.setSpacing(1)
        for tag, t in PALETTE:
            if t not in NODES:
                continue
            b = QToolButton()
            b.setIcon(tag_icon(tag, title_color(t, NODES[t].category)))
            b.setIconSize(QSize(24, 20))
            b.setToolTip(NODES[t].label)
            b.clicked.connect(lambda _=False, t=t: self.add_node(t, self.view.mapToScene(self.view.viewport().rect().center())))
            lay.addWidget(b)
        lay.addStretch(1)
        lay.addWidget(QLabel("Parent size:"))
        self.parent_w, self.parent_h = QComboBox(), QComboBox()
        for c in (self.parent_w, self.parent_h):
            c.addItems(SIZES)
            c.currentTextChanged.connect(self._parent_size)
        self.parent_link = QToolButton()
        self.parent_link.setIcon(icon("chain"))
        self.parent_link.setCheckable(True)
        self.parent_link.setChecked(True)
        self.parent_link.setToolTip("Link width and height")
        lay.addWidget(self.parent_w)
        lay.addWidget(self.parent_link)
        lay.addWidget(self.parent_h)
        reset = QToolButton()
        reset.setIcon(icon("undo"))
        reset.setToolTip("Reset parent size to 256 x 256")
        reset.clicked.connect(lambda: self._set_parent_size(8, 8))
        lay.addWidget(reset)
        return w

    # --- tabs ---------------------------------------------------------------------------------

    @property
    def graph(self):
        return self.package.graphs.get(self.graph_id) if self.package else None

    def open_graph(self, package, graph_id: str) -> None:
        for i, (p, g) in enumerate(self.tabs_data):
            if p is package and g == graph_id:
                self.tabbar.setCurrentIndex(i)
                return
        self.tabs_data.append((package, graph_id))
        self.tabbar.addTab(icon("graph"), graph_id)
        self.tabbar.setCurrentIndex(len(self.tabs_data) - 1)
        self._tab_changed(len(self.tabs_data) - 1)  # the first addTab doesn't emit currentChanged

    def close_tab(self, index: int) -> None:
        self.tabs_data.pop(index)
        self.tabbar.removeTab(index)
        if not self.tabs_data:
            self.package, self.graph_id = None, ""
            self.scene.clear()
            self.title_changed.emit("Graph")

    def close_package(self, package) -> None:
        for i in reversed(range(len(self.tabs_data))):
            if self.tabs_data[i][0] is package:
                self.close_tab(i)

    def open_function(self, node_id: str, param: str, is_param_function: bool) -> None:
        """Edit a node's function graph (a FUNCTION parameter, or the function driving a parameter)."""
        self.open_function_target(NodeFunction(self.package, self.graph_id, node_id, param, is_param_function))

    def open_function_target(self, target) -> None:
        self._function_from_fx = isinstance(target, FxFunction)
        self.functions.open(target)
        self.pages.setCurrentWidget(self.functions)

    def close_function(self) -> None:
        if self._function_from_fx:  # back to the FX-Map the function belongs to
            self.pages.setCurrentWidget(self.fxmaps)
            self.fxmaps.sync()
            return
        self.pages.setCurrentIndex(0)
        self.sync()

    # --- graph items, disabled nodes, pins --------------------------------------------------------

    def add_item(self, kind: str, pos: QPointF):
        self.edit(f"Add {kind}", lambda: self.graph.add_item(kind, (pos.x(), pos.y())))

    def update_item(self, item_id, position, size, moved_nodes, moved_items):
        g = self.graph

        def do():
            item = g.item(item_id)
            item["position"] = list(position)
            if size is not None and "size" in item:
                item["size"] = list(size)
            for k, (x, y) in moved_nodes.items():
                if k in g.nodes:
                    g.nodes[k].position = [x, y]
            for i in g.items:
                if i["id"] in moved_items:
                    i["position"] = list(moved_items[i["id"]])
        self.edit("Move item", do, layout=True)

    def remove_items(self, ids):
        self.edit("Delete items", lambda: [self.graph.remove_item(i) for i in ids])

    def edit_item_text(self, item_id):
        from PySide6.QtWidgets import QInputDialog
        item = self.graph.item(item_id)
        field = "text" if item["kind"] == "comment" else "title"
        if item["kind"] == "comment":
            text, ok = QInputDialog.getMultiLineText(self, "Comment", "Text", item.get("text", ""))
        else:
            text, ok = QInputDialog.getText(self, item["kind"].capitalize(), "Title", text=item.get("title", ""))
        if ok:
            self.edit("Edit text", lambda: self.graph.item(item_id).__setitem__(field, text))

    def toggle_disabled(self):
        ids = self.scene.selected_ids()
        if not ids or self.graph is None:
            return
        state = not all(self.graph.nodes[k].disabled for k in ids)
        self.edit("Disable node" if state else "Enable node", lambda: self.graph.set_disabled(ids, state))

    def go_to_pin(self, step: int):
        pins = [i for i in (self.graph.items if self.graph else []) if i["kind"] == "pin"]
        if not pins:
            return
        self._pin_index = (getattr(self, "_pin_index", -1) + step) % len(pins)
        x, y = pins[self._pin_index]["position"]
        self.view.centerOn(QPointF(x, y))

    def edit_inner_graph(self) -> None:
        """Ctrl+E: the selected node's FX-Map graph or function (Pixel/Value processor)."""
        ids = self.scene.selected_ids()
        if len(ids) != 1 or self.graph is None:
            return
        t = self.graph.nodes[ids[0]].type
        if t == "stamp_map":
            self.open_fxmap(ids[0])
        elif t in ("pixel_processor", "value_processor"):
            self.open_function(ids[0], "per_pixel_function" if t == "pixel_processor" else "value_processor_function",
                               False)

    def open_fxmap(self, node_id: str, param: str = "fxmap") -> None:
        self.fxmaps.open(self.package, self.graph_id, node_id, param)
        self.pages.setCurrentWidget(self.fxmaps)

    def _tab_changed(self, index: int) -> None:
        self.pages.setCurrentIndex(0)
        if index < 0 or index >= len(self.tabs_data):
            return
        self.package, self.graph_id = self.tabs_data[index]
        self.scene.clear()
        self.sync()
        QTimer.singleShot(0, self.view.focus)
        self.graph_shown.emit(self.package, self.graph_id)

    def sync(self, thumbnails: bool = True) -> None:
        g = self.graph
        if g is None:
            return
        self.scene.sync(g, self.package)
        if not self.doc.dragging:  # its warnings at the release of a dragged value, not at each step
            self._show_warnings(g)
        for c, v in ((self.parent_w, g.parent_size[0]), (self.parent_h, g.parent_size[1])):
            c.blockSignals(True)
            c.setCurrentText(str(1 << v))
            c.blockSignals(False)
        self.title_changed.emit(f"{self.graph_id} - Graph")
        from trafilatura3d.editor import preferences as prefs
        if thumbnails and prefs.get("auto_thumbnails"):  # Preferences: all nodes thumbnails when opening a graph
            self.thumb_timer.start()
        self._apply_flow()

    def apply_preferences(self):
        from trafilatura3d.editor import preferences as prefs
        self.tab_shortcut.setEnabled(prefs.get("tab_node_menu"))
        self.drag_connector_menu = prefs.get("drag_connector_menu")
        for sc in self.node_shortcuts:
            sc.setParent(None)
        self.node_shortcuts = []
        for node_type, keys in prefs.shortcuts().items():
            self.node_shortcuts += shortcut(keys, self.view, lambda t=node_type: self.add_node(t, self.view.cursor_scene_pos),
                                            Qt.WidgetWithChildrenShortcut)

    def _show_warnings(self, g):
        """A graph's warnings are listed in the bottom-left corner of the Graph view, and its
        nodes with warnings show a badge."""
        from trafilatura3d.diagnostics import graph_warnings
        try:
            found = graph_warnings(self.package, g.id)
        except Exception as e:  # noqa: BLE001 - diagnostics must never break the view
            found = []
            self.warnings_label.setText(f"⚠ warnings unavailable: {e}")
        by_node = {}
        for w in found:
            if w.node:
                by_node.setdefault(w.node, []).append(w.message + (f" ({w.detail})" if w.detail else ""))
        for nid, item in self.scene.nodes.items():
            item.warnings = by_node.get(nid, [])
            item.setToolTip(f"{item.label}\n{nid}" + "".join(f"\n⚠ {m}" for m in item.warnings))
            item.update()
        if found:
            missing = {nid for nid, item in self.scene.nodes.items() if item.missing}
            found = sorted(found, key=lambda w: w.node not in missing)  # missing nodes first: they break the most
            lines = [html.escape(f"⚠ {w.message}") + (f'  <a href="{html.escape(w.node)}" style="color:#f0c070">'
                                                     f"[{html.escape(w.node)}]</a>" if w.node else "")
                     for w in found[:8]]
            if len(found) > 8:
                lines.append(f"… {len(found) - 8} more")
            shown = frozenset((w.message, w.node) for w in found)
            if self.dismissed_warnings.get((id(self.package), g.id)) == shown:
                self.warnings_label.hide()
                return
            self._shown_warnings = shown
            lines[0] = '<a href="#dismiss" style="color:#f0c070; text-decoration:none">✕</a>&nbsp;&nbsp;' + lines[0]
            self.warnings_label.setText("<br>".join(lines))
            self.warnings_label.adjustSize()
            self.warnings_label.show()
            self._place_warnings()
        else:
            self.warnings_label.hide()

    def _warning_link(self, target: str) -> None:
        """The warnings' links: ✕ closes them for this graph (they come back when they change), a node frames it."""
        if target == "#dismiss":
            self.dismissed_warnings[(id(self.package), self.graph_id)] = self._shown_warnings
            self.warnings_label.hide()
        else:
            self._focus_node(target)

    def _focus_node(self, node_id: str) -> None:
        item = self.scene.nodes.get(node_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.view.focus()

    def _place_warnings(self):
        lab = self.warnings_label
        lab.move(8, self.view.height() - lab.height() - 8)

    def _doc_changed(self, package) -> None:
        if package is not self.package:
            return
        for i in reversed(range(len(self.tabs_data))):  # graphs removed by the edit
            p, g = self.tabs_data[i]
            if p is package and g not in package.graphs:
                self.close_tab(i)
        if self.graph is not None:
            self.sync(thumbnails=False)  # the edit's thumbnails: steps of the Graph workspace's `LiveCook`

    def _layout_changed(self, package) -> None:
        """Only positions changed (undo/redo of a move, a remote edit): the scene follows, nothing else does."""
        if package is self.package and self.graph is not None and not self._arranging:
            self.scene.sync(self.graph, self.package)

    # --- thumbnails ---------------------------------------------------------------------------

    def refresh_thumbnails(self) -> None:
        """Every stale thumbnail now (Graph view > Compute node thumbnails)."""
        run(self.thumbnail_steps(self.stale_thumbnails()))

    def _schedule_thumbnails(self) -> None:
        self.doc.live.schedule()

    def stale_thumbnails(self) -> list:
        """(node id, job, output) of the nodes whose thumbnail is missing or shows another result, inputs first."""
        compiled = self.doc.compiled(self.package, self.graph_id) if self.graph is not None else None
        if compiled is None:
            return []
        stale = []
        for node_id, outs in compiled.nodes.items():
            if outs:
                job, out = next(iter(outs.values()))
                item = self.scene.nodes.get(node_id)
                if item is not None and item.thumbnail is not None and item.thumbnail_key == (job.key, out):
                    continue  # the result it shows hasn't changed (a moved node, an edit elsewhere)
                stale.append((node_id, job, out))
        return stale

    def thumbnail_steps(self, stale: list, related=(), precious: frozenset = frozenset()):
        """The thumbnails of `stale_thumbnails` (or some of them) computed and shown, a step per wait of the cook
        (`Evaluator.cook_steps`); `related`, `precious`: the refresh's (`LiveCook`), kept by the budget."""
        if not stale:
            return
        items = [(job, out) for _, job, out in stale]
        try:
            images = yield from self.doc.ev.thumbnail_steps(items, 128, related=related, precious=precious)
        except Exception as e:  # a failing node must not stop the editor
            self.status(f"Cook error: {e}")
            return
        for (node_id, job, out), img in zip(stale, images):
            q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 4, QImage.Format_RGBA8888).copy()
            tag = ("L" if job.is_gray(out) else "C") + DEPTH_TAG[job.depths[out]]
            self.scene.set_thumbnail(node_id, q, f"{job.size[0]}x{job.size[1]} - {tag}", (job.key, out))
        self._show_timings()

    # --- timings --------------------------------------------------------------------------------

    def _timings_toggled(self, on: bool) -> None:
        if on and not self._confirm_timings():
            self.act_timings.blockSignals(True)
            self.act_timings.setChecked(False)
            self.act_timings.blockSignals(False)
            return
        self.doc.ev.timing = on
        self._info_changed()
        if on:
            self.reset_timings()

    def _confirm_timings(self) -> bool:
        """Timings slow the cooks down: said before they are turned on."""
        from PySide6.QtWidgets import QMessageBox
        how = ("Each node gets GPU timestamps: cooking is a little slower." if self.doc.ev.timestamps_work() else
               "This GPU has no usable timestamps: each node is cooked and waited for on its own, and cooking gets "
               "much slower (two to three times on a graph of light nodes).")
        box = QMessageBox(QMessageBox.Warning, "Display timings",
                          f"Displaying timings measures every node that cooks. {how}\n\n"
                          "Use it to find the slow nodes, then turn it off.",
                          QMessageBox.Ok | QMessageBox.Cancel, self)
        box.setDefaultButton(QMessageBox.Cancel)
        return box.exec() == QMessageBox.Ok

    def reset_timings(self) -> None:
        """Every node of the graph cooked again and timed (the timings of the first cook of a node type include
        compiling its shaders)."""
        compiled = self.doc.compiled(self.package, self.graph_id) if self.graph is not None else None
        if compiled is None:
            return
        ev = self.doc.ev
        jobs = [job for outs in compiled.nodes.values() for job, _ in outs.values()]
        for job in jobs:
            ev.timings.pop(job.key, None)
        ev.forget(jobs)
        for item in self.scene.nodes.values():
            item.thumbnail_key = None  # cooked again with its thumbnail
        self.refresh_thumbnails()

    def _show_timings(self) -> None:
        """Each node's last cook time under its size line (a node computed inside the next one's dispatch has
        none)."""
        compiled = self.doc.compiled(self.package, self.graph_id) if self.graph is not None else None
        if compiled is None:
            return
        timings = self.doc.ev.timings
        for node_id, outs in compiled.nodes.items():
            item = self.scene.nodes.get(node_id)
            if item is None or not outs:
                continue
            ms = timings.get(next(iter(outs.values()))[0].key)
            if ms != item.timing_ms:
                item.timing_ms = ms
                item.update()

    # --- controller: called by the scene --------------------------------------------------------

    def node_label(self, node, definition) -> str:
        if node.type == "output" or node.type in INPUT_TYPES:
            v = definition.values(node)
            return v["label"] or v["identifier"]
        return definition.label

    def edit(self, label: str, fn, merge: str | None = None, layout: bool = False):
        """One undoable edit; a `layout` edit is a move made in this view (nothing recooks, the scene already shows
        it)."""
        self._arranging = layout
        try:
            return self.doc.edit(self.package, label, fn, merge, layout)
        finally:
            self._arranging = False

    def connect(self, src, out, dst, port):
        """Connect; with Preferences > Automatically insert color/greyscale conversion node, a grayscale
        output into a color input goes through a Gradient Map, a color one into a grayscale input
        through a Grayscale Conversion."""
        from trafilatura3d.editor import preferences as prefs
        from trafilatura3d.nodes.base import PortType
        g = self.graph
        got, want = g.output_type(src, out), g.definition(dst).inputs[port].type
        conv = None
        if prefs.get("auto_conversion") and not want.accepts(got):
            conv = {(PortType.GRAY, PortType.COLOR): ("gradient_map", "input"),
                    (PortType.COLOR, PortType.GRAY): ("grayscale_conversion", "input")}.get((got, want))
        if conv is None:
            self.edit("Connect", lambda: g.connect(src, out, dst, port))
            return

        def do():
            a, b = g.nodes[src].position, g.nodes[dst].position
            mid = g.add_node(conv[0], ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            g.connect(src, out, mid, conv[1])
            g.connect(mid, next(iter(g.definition(mid).outputs)), dst, port)
        self.edit("Connect with conversion", do)

    def disconnect_port(self, node_id, port, is_input):
        g = self.graph
        if is_input:
            self.edit("Disconnect", lambda: g.disconnect(node_id, port))
        else:
            links = [(n.id, p) for n in g.nodes.values() for p, (s, o) in n.inputs.items() if s == node_id and o == port]
            self.remove_links(links)

    def insert_dot(self, dst, port, pos: QPointF) -> str | None:
        """Alt+click on a link: a Dot centred there, the link going through it (one undo step)."""
        g = self.graph
        half = graphtools.DOT / 2
        return self.edit("Insert dot", lambda: graphtools.insert_dot(g, dst, port, (pos.x() - half, pos.y() - half)))

    def remove_links(self, links):
        g = self.graph
        self.edit("Remove links", lambda: [g.disconnect(d, p) for d, p in links])

    def relink(self, links, anchor, target, keep):
        g = self.graph

        def do():
            for link in links:
                src, dst = (link.src.node.node_id, link.src.name), (link.dst.node.node_id, link.dst.name)
                if not keep:
                    g.disconnect(*dst)
                if anchor.is_input:
                    g.connect(src[0], src[1], target.node.node_id, target.name)
                else:
                    g.connect(target.node.node_id, target.name, *dst)
        self.edit("Move links" if not keep else "Duplicate links", do)

    def move_nodes(self, positions: dict):
        g = self.graph

        def do():
            for k, (x, y) in positions.items():
                if k in g.nodes:
                    g.nodes[k].position = [x, y]
        self.edit("Move nodes", do, layout=True)

    def add_node(self, node_type, pos: QPointF, connect_from=None, ref: str = "", params: dict | None = None) -> str | None:
        g = self.graph
        if g is None:
            return None
        selected = self.scene.selected_ids()

        def do():
            new = graphtools.add_node(g, node_type, (pos.x(), pos.y()), ref, params)  # + Project settings defaults
            d = g.definition(new)
            port = d.primary_input()
            src = connect_from
            if src is None and len(selected) == 1 and g.definition(selected[0]).outputs:  # auto-link
                src = (selected[0], next(iter(g.definition(selected[0]).outputs)))
            if src is not None and port is not None:
                try:
                    g.connect(src[0], src[1], new, port)
                except DocumentError:
                    pass
            return new
        new = self.edit(f"Add {node_type}", do)
        if new:
            self.scene.clearSelection()
            if new in self.scene.nodes:
                self.scene.nodes[new].setSelected(True)
            self.node_created.emit(self.package, self.graph_id, new)
        return new

    def delete(self, nodes, links, relink=False):
        g = self.graph

        def do():
            for dst, port in links:
                if dst not in nodes:
                    g.disconnect(dst, port)
            for node_id in nodes:
                if relink:
                    self._relink_around(node_id, set(nodes))
                g.remove_node(node_id)
        self.edit("Delete and relink" if relink else "Delete", do)

    def _relink_around(self, node_id, removing):
        """Connect what fed the node's primary input to what its outputs fed, where types allow."""
        g = self.graph
        d = g.definition(node_id)
        port = d.primary_input()
        if port is None or port not in g.nodes[node_id].inputs:
            return
        src = g.nodes[node_id].inputs[port]
        for n in list(g.nodes.values()):
            if n.id in removing:
                continue
            for p, (s, o) in list(n.inputs.items()):
                if s == node_id:
                    try:
                        g.connect(src[0], src[1], n.id, p)
                    except DocumentError:
                        g.disconnect(n.id, p)

    def copy(self):
        g = self.graph
        ids = set(self.scene.selected_ids())
        if not ids:
            return
        nodes = []
        for k in ids:
            d = g.nodes[k].to_dict()
            d["inputs"] = {p: v for p, v in d.get("inputs", {}).items() if v[0] in ids}
            nodes.append(d)
        self.clipboard = nodes
        QApplication.clipboard().setText(json.dumps({"trafilatura3d_nodes": nodes}))

    def paste(self, pos: QPointF, nodes=None):
        nodes = nodes or self.clipboard
        if not nodes or self.graph is None:
            return
        g = self.graph
        x0 = min(n["position"][0] for n in nodes)
        y0 = min(n["position"][1] for n in nodes)

        def do():
            mapping = {}
            for n in nodes:
                new = g.add_node(n["type"], (pos.x() + n["position"][0] - x0, pos.y() + n["position"][1] - y0),
                                 ref=n.get("ref", ""))
                mapping[n["id"]] = new
                for k, v in n.get("params", {}).items():
                    if k != "identifier":
                        g.set_param(new, k, v)
                g.nodes[new].base = dict(n.get("base", {}))
            for n in nodes:
                for port, (src, out) in n.get("inputs", {}).items():
                    g.connect(mapping[src], out, mapping[n["id"]], port)
            return list(mapping.values())
        new = self.edit("Paste", do)
        if new:
            self.scene.clearSelection()
            for k in new:
                if k in self.scene.nodes:
                    self.scene.nodes[k].setSelected(True)

    def duplicate(self):
        self.copy()
        if self.clipboard:
            x0 = min(n["position"][0] for n in self.clipboard)
            y0 = min(n["position"][1] for n in self.clipboard)
            self.paste(QPointF(x0 + 32, y0 + 32))

    def open_node_menu(self, scene_pos: QPointF, from_port=None):
        if self.graph is None:
            return
        accepts = None
        if from_port is not None:
            accepts = self.graph.output_type(*from_port)
        elif len(self.scene.selected_ids()) == 1:
            sel = self.scene.selected_ids()[0]
            outs = self.graph.definition(sel).outputs
            if outs:
                from_port = (sel, next(iter(outs)))
                accepts = self.graph.output_type(*from_port)
        menu = NodeMenu(self, accepts)
        menu.chosen.connect(lambda t: self.add_node(t, scene_pos, from_port))
        menu.move(self.view.mapToGlobal(self.view.mapFromScene(scene_pos)))
        menu.show()

    def view_2d_node(self, node_id):
        self.view_2d.emit(self.package, self.graph_id, node_id)

    def fill_context_menu(self, menu: QMenu, node_id, pos: QPointF):
        g = self.graph
        if g is None:
            return
        if node_id is not None:
            if node_id not in self.scene.selected_ids():
                self.scene.clearSelection()
                self.scene.nodes[node_id].setSelected(True)
            menu.addAction("View output in 2D view", lambda: self.view_2d.emit(self.package, self.graph_id, node_id))
            v3 = menu.addMenu("View in 3D view")
            v3.addAction("Preview (this output alone, unlit)",
                         lambda: self.view_3d_node.emit(self.package, self.graph_id, node_id, None))
            assign = v3.addMenu("Assign to channel")
            from trafilatura3d.editor.view3d import SLOTS
            for slot in SLOTS:
                assign.addAction(slot, lambda s=slot: self.view_3d_node.emit(self.package, self.graph_id, node_id, s))
            menu.addAction("Reset and view outputs in 3D view",
                           lambda: self.view_3d_reset.emit(self.package, self.graph_id))
            menu.addSeparator()
            menu.addAction("Copy info to clipboard", lambda: QApplication.clipboard().setText(node_id))
            n = g.nodes[node_id]
            if n.type == INSTANCE:
                menu.addAction("Open reference", lambda: self.open_reference(n.ref))
            if n.type == "stamp_map":
                menu.addAction("Edit Stamp Map graph (Ctrl+E)", lambda: self.open_fxmap(node_id))
            if n.type in ("pixel_processor", "value_processor"):
                param = "per_pixel_function" if n.type == "pixel_processor" else "value_processor_function"
                menu.addAction("Edit function (Ctrl+E)", lambda: self.open_function(node_id, param, False))
            menu.addSeparator()
            menu.addAction("Copy selection", self.copy)
            menu.addAction("Delete selection", lambda: self.delete(self.scene.selected_ids(), []))
            menu.addAction("Delete and relink", lambda: self.delete(self.scene.selected_ids(), [], relink=True))
            menu.addAction("Duplicate selection", self.duplicate)
            menu.addSeparator()
            menu.addAction("Select upstream nodes", lambda: self.scene.select_stream(node_id, True))
            menu.addAction("Select downstream nodes", lambda: self.scene.select_stream(node_id, False))
            menu.addSeparator()
            dis = menu.addAction("Disable node/selection (Shift+D)", self.toggle_disabled)
            dis.setEnabled(all(g.can_disable(k) for k in self.scene.selected_ids()))
            return
        add = menu.addMenu("Add node")
        menu.addAction("Add comment", lambda: self.add_item("comment", pos))
        menu.addAction("Add frame", lambda: self.add_item("frame", pos))
        menu.addAction("Add pin", lambda: self.add_item("pin", pos))
        menu.addAction("Add Dot node", lambda: self.add_node("dot", pos))
        menu.addAction("Previous pin (Shift+F2)", lambda: self.go_to_pin(-1))
        menu.addAction("Next pin (F2)", lambda: self.go_to_pin(1))
        menu.addSeparator()
        by_cat = {}
        for label, category, t in library_entries():
            by_cat.setdefault(category, []).append((label, t))
        for category in sorted(by_cat):
            sub = add
            for part in category.split("/"):
                found = next((a.menu() for a in sub.actions() if a.menu() and a.text() == part), None)
                sub = found or sub.addMenu(part)
            for label, t in by_cat[category]:
                action = sub.addAction(label, lambda t=t: self.add_node(t, pos))
                if (i := library_icon(t)) is not None:
                    action.setIcon(i)
        menu.addSeparator()
        menu.addAction("View outputs in 3D view", lambda: self.view_3d.emit(self.package, self.graph_id))
        menu.addAction("Reset and view outputs in 3D view", lambda: self.view_3d_reset.emit(self.package, self.graph_id))
        menu.addAction("Compute node thumbnails", self.refresh_thumbnails)
        menu.addAction("Clear nodes thumbnails", self.clear_thumbnails)
        menu.addSeparator()
        paste = menu.addAction("Paste", lambda: self.paste(pos))
        paste.setEnabled(bool(self.clipboard))
        menu.addAction("Select all", lambda: [i.setSelected(True) for i in self.scene.nodes.values()])
        if self.scene.selected_ids():
            menu.addSeparator()
            menu.addAction("Copy selection", self.copy)
            menu.addAction("Delete selection", lambda: self.delete(self.scene.selected_ids(), []))
            menu.addAction("Delete and relink", lambda: self.delete(self.scene.selected_ids(), [], relink=True))
            menu.addAction("Duplicate selection", self.duplicate)

    def open_reference(self, ref: str):
        try:
            pkg, graph = self.package.resolve(ref)
        except DocumentError as e:
            self.status(str(e))
            return
        if pkg not in self.doc.packages:
            self.doc.packages.append(pkg)
            self.doc.packages_changed.emit()
        self.open_graph(pkg, graph.id)

    def _drop(self, mime, pos: QPointF) -> bool:
        if self.graph is None:
            return False
        if mime.hasFormat("application/x-trafilatura3d-node"):
            types = bytes(mime.data("application/x-trafilatura3d-node")).decode().split("\n")
            for i, t in enumerate(types):
                at = QPointF(pos.x() + i * 128, pos.y())
                if t.startswith("graph:"):  # a graph of the project's Library: an instance of it
                    import os
                    path, gid = t[6:].rsplit("#", 1)
                    if self.package.path is not None:
                        try:
                            path = os.path.relpath(path, self.package.path.parent).replace(os.sep, "/")
                        except ValueError:
                            pass
                    self.add_node("graph_instance", at, ref=f"{path}#{gid}")
                else:
                    self.add_node(t, at)
            return True
        if mime.hasFormat("application/x-trafilatura3d-graph"):
            pkg_index, gid = bytes(mime.data("application/x-trafilatura3d-graph")).decode().split("\n")
            pkg = self.doc.packages[int(pkg_index)]
            if pkg is self.package:
                ref = gid
            elif pkg.path is None or self.package.path is None:
                self.status("Save both packages to instance a graph from another package")
                return True
            else:
                import os
                ref = os.path.relpath(pkg.path, self.package.path.parent).replace(os.sep, "/") + "#" + gid
            if pkg is self.package and gid == self.graph_id:
                self.status("A graph can't contain an instance of itself")
                return True
            self.add_node(INSTANCE, pos, ref=ref)
            return True
        if mime.hasFormat("application/x-trafilatura3d-resource"):
            pkg_index, rid = bytes(mime.data("application/x-trafilatura3d-resource")).decode().split("\n")
            pkg = self.doc.packages[int(pkg_index)]
            r = pkg.resources.get(rid)
            if r is None or r.kind != "bitmap":
                self.status("Only bitmap resources make a node: a 3D scene opens with a double click")
                return True
            path = r.path
            if pkg is not self.package:  # relative to the graph's package
                import os
                if pkg.path is None or self.package.path is None:
                    self.status("Save both packages to use a bitmap from another package")
                    return True
                path = os.path.relpath(pkg.path.parent / r.path, self.package.path.parent).replace(os.sep, "/")
            self.add_node("bitmap", pos, params={"pkg_resource_path": path})
            return True
        return False

    def clear_thumbnails(self):
        for item in self.scene.nodes.values():
            item.thumbnail = None
            item.update()

    # --- toolbar actions ----------------------------------------------------------------------

    def _selection(self):
        ids = self.scene.selected_ids()
        self.selection_changed.emit(self.package, self.graph_id, ids[0] if len(ids) == 1 else None)
        self._apply_flow()

    def _info_changed(self):
        sc = self.scene
        sc.show_names, sc.show_info = self.act_names.isChecked(), self.act_size.isChecked()
        sc.show_badges, sc.limit_text = self.act_badges.isChecked(), self.act_limit.isChecked()
        sc.show_timing = self.act_timings.isChecked()
        for item in sc.nodes.values():
            item.prepareGeometryChange()
            item.show_names, item.show_info = sc.show_names, sc.show_info
            item.show_badges, item.limit_text, item.show_timing = sc.show_badges, sc.limit_text, sc.show_timing
            item.update()
            for port in list(item.inputs.values()) + list(item.outputs.values()):
                port.prepareGeometryChange()
                port.update()

    def _flow(self, on):
        self.highlight_flow = on
        self._apply_flow()

    def _apply_flow(self):
        g = self.graph
        ids = set(self.scene.selected_ids())
        related = set()
        if self.highlight_flow and g is not None and ids:
            for k in ids:
                if k in g.nodes:
                    related |= set(g.upstream([k])) | g.downstream(k)
        for link in self.scene.links:
            on = link.src.node.node_id in related and link.dst.node.node_id in related
            if link.highlight != on:
                link.highlight = on
                link.update()

    def _rect_links(self, on):
        self.scene.rectangular = on
        for link in self.scene.links:
            link.rectangular = on
            link.update_path()

    def _parent_size(self, _text):
        w = int(self.parent_w.currentText()).bit_length() - 1
        h = int(self.parent_h.currentText()).bit_length() - 1
        if self.parent_link.isChecked():
            h = w if self.sender() is self.parent_w else h
            w = h if self.sender() is self.parent_h else w
        self._set_parent_size(w, h)

    def _set_parent_size(self, w, h):
        g = self.graph
        if g is not None and g.parent_size != [w, h]:
            self.edit("Parent size", lambda: setattr(g, "parent_size", [w, h]))

    def align(self, mode):
        items = [self.scene.nodes[k] for k in self.scene.selected_ids()]
        if len(items) < 2:
            return
        xs = [i.pos().x() for i in items]
        ys = [i.pos().y() for i in items]
        pos = {}
        if mode in ("left", "hcenter", "right"):
            x = {"left": min(xs), "right": max(xs), "hcenter": sum(xs) / len(xs)}[mode]
            pos = {i.node_id: (x, i.pos().y()) for i in items}
        elif mode in ("top", "vcenter", "bottom"):
            y = {"top": min(ys), "bottom": max(ys), "vcenter": sum(ys) / len(ys)}[mode]
            pos = {i.node_id: (i.pos().x(), y) for i in items}
        elif mode == "hdist":
            items.sort(key=lambda i: i.pos().x())
            step = (max(xs) - min(xs)) / (len(items) - 1)
            pos = {i.node_id: (min(xs) + k * step, i.pos().y()) for k, i in enumerate(items)}
        else:
            items.sort(key=lambda i: i.pos().y())
            step = (max(ys) - min(ys)) / (len(items) - 1)
            pos = {i.node_id: (i.pos().x(), min(ys) + k * step) for k, i in enumerate(items)}
        self.move_nodes(pos)

    def export_view(self):
        if not self.scene.nodes:
            return
        name, _ = QFileDialog.getSaveFileName(self, "Export Graph view", f"{self.graph_id}.png", "PNG (*.png)")
        if not name:
            return
        rect = self.scene.itemsBoundingRect().adjusted(-40, -40, 40, 40)
        img = QImage(int(rect.width()), int(rect.height()), QImage.Format_ARGB32)
        img.fill(QColor("#3a3a3a"))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        self.scene.render(p, source=rect)
        p.end()
        img.save(name)

    def find_node(self):
        if self.graph is None:
            return
        menu = QMenu(self)
        for k in sorted(self.scene.nodes):
            item = self.scene.nodes[k]
            menu.addAction(f"{item.label}  ({k})", lambda k=k: self._go_to(k))
        menu.exec(self.view.mapToGlobal(self.view.rect().center()))

    def _go_to(self, node_id):
        self.scene.clearSelection()
        self.scene.nodes[node_id].setSelected(True)
        self.view.focus()
