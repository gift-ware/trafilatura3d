"""The graph scene: draws a graph and turns gestures into edits through its controller (it never
changes the document itself). Gestures of the Graph view:

- connect: drag from a connector to another, or click one then the other; Shift+drag from a
  connector moves its links, Ctrl+drag duplicates them; Alt+click removes all its links;
- Alt over a link shows where a Dot would go, Alt+click puts it there (the link goes through it);
- Delete removes the selection (breaking links), Backspace removes it and relinks across;
- Ctrl+C / Ctrl+V / Ctrl+D copy, paste (at the cursor), duplicate; Ctrl+A selects all;
- right click: the empty-space or node context menu; Space opens the node menu;
- nodes snap to the grid while moved (hold Ctrl to move freely).
"""

from __future__ import annotations

import copy

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsScene, QMenu

from trafilatura3d.package import DocumentError
from trafilatura3d.editor.graph.items import (DotItem, GraphItem, LinkItem, NodeItem, PortItem, link_path,
                                              port_color)
from trafilatura3d.graphtools import DOT, DOT_TYPE

GRID = 16
SCENE = QRectF(-50000, -50000, 100000, 100000)  # the least the view can scroll over
SCENE_MARGIN = 20000  # room to scroll past the farthest items


class GraphScene(QGraphicsScene):
    def __init__(self, controller):
        super().__init__()
        self.ctl = controller
        self.nodes: dict[str, NodeItem] = {}
        self.links: list[LinkItem] = []
        self.rectangular = False
        # Node information menu: connector names, node size, badges, text scaling
        self.show_names = True
        self.show_info = True
        self.show_badges = True
        self.show_timing = False
        self.limit_text = True
        self._drag = None  # (anchor port, temp path, mode, moved links)
        self._pending = None  # first connector of a click-click connection
        self._syncing = False
        self._dot_preview = None  # the Dot an Alt+click on a link would add
        self.setSceneRect(SCENE)

    def clear(self) -> None:
        """Every item removed, and what the scene keeps of them (a kept reference to a deleted item fails at the
        next mouse move)."""
        super().clear()
        self.nodes, self.links, self.graph_items = {}, [], []
        self._drag = self._pending = self._dot_preview = None

    # --- sync ---------------------------------------------------------------------------------

    def sync(self, graph, package) -> None:
        """Make the items match the graph (positions, ports, links), keeping the selection."""
        self._syncing = True
        selected = self.selected_ids()
        selected_items = {i.item_id for i in self.selectedItems() if isinstance(i, GraphItem)}
        for node_id in list(self.nodes):
            if node_id not in graph.nodes:
                self.removeItem(self.nodes.pop(node_id))
        missing = set()
        types_memo: dict = {}  # the output types resolved so far, shared by every node
        for n in graph.nodes.values():
            pretty = lambda k: k.replace("_", " ").capitalize()  # identifiers without a label
            try:
                d = package.definition(n)
            except (DocumentError, OSError, ValueError):
                # A graph instance whose package or graph is missing: a placeholder with the outputs it
                # declares (or its links name) and its inputs, so the links stay visible, drawn in red.
                d = None
                outs = list(n.outputs) or sorted({out for m in graph.nodes.values()
                                                  for src, out in m.inputs.values() if src == n.id})
                ports_in = [(k, pretty(k)) for k in n.inputs]
                ports_out = [(k, pretty(k)) for k in outs] or [("output", "Output")]
                label = n.ref.rsplit('#', 1)[-1] or n.type  # the red title and the Missing badge say the rest
                missing.add(n.id)
            if d is not None:
                shown = set(d.visible_inputs(d.values(n)))  # numbered inputs beyond the count are hidden
                ports_in = [(k, v.label or pretty(k)) for k, v in d.inputs.items() if k in shown]
                shown_out = set(d.visible_outputs(d.values(n)))  # instance outputs hidden by Visible if
                ports_out = [(k, v.label or pretty(k)) for k, v in d.outputs.items() if k in shown_out]
                label = self.ctl.node_label(n, d)
            item = self.nodes.get(n.id)
            if item is None or [p for p, _ in ports_in] != list(item.inputs) \
                    or [p for p, _ in ports_out] != list(item.outputs) or item.label != label:
                thumb = item.thumbnail if item else None
                if item is not None:
                    self.removeItem(item)
                kind = DotItem if n.type == DOT_TYPE else NodeItem
                item = kind(n.id, label, n.type, d.category if d else "Graph instances", ports_in, ports_out)
                item.thumbnail = thumb
                self.nodes[n.id] = item
                self.addItem(item)
            item.show_names, item.show_info = self.show_names, self.show_info
            item.show_badges, item.limit_text, item.show_timing = self.show_badges, self.limit_text, self.show_timing
            item.setPos(*n.position)
            item.setSelected(n.id in selected)
            given = graph.input_types(n.id, types_memo) if d else {}  # a placeholder's connectors have no type
            types = d.output_types(given, d.values(n)) if d else {}
            for name, port in item.outputs.items():
                port.type = types.get(name)
                port.connected = False
            for name, port in item.inputs.items():
                port.connected = name in n.inputs
                port.type = (given.get(name) or d.inputs[name].type) if d else None
        for n in graph.nodes.values():
            self.nodes[n.id].disabled = n.disabled
            self.nodes[n.id].missing = n.id in missing
        for gi in getattr(self, "graph_items", []):
            self.removeItem(gi)
        self.graph_items = []
        for data in getattr(graph, "items", []):
            gi = GraphItem(copy.deepcopy(data))  # a copy: the document changes only through edits
            gi.setSelected(data["id"] in selected_items)
            self.addItem(gi)
            self.graph_items.append(gi)
        for link in self.links:
            self.removeItem(link)
        self.links = []
        for n in graph.nodes.values():
            for port, (src, out) in n.inputs.items():
                if src in self.nodes and port in self.nodes[n.id].inputs \
                        and out in self.nodes[src].outputs:  # hidden connectors keep their link
                    s = self.nodes[src].port(out, False)
                    s.connected = True
                    link = LinkItem(s, self.nodes[n.id].port(port, True))
                    link.rectangular = self.rectangular
                    link.update_path()
                    self.addItem(link)
                    self.links.append(link)
        self._syncing = False
        self.fit_rect()
        self.update()

    def fit_rect(self) -> None:
        """The scrollable area: SCENE grown to the items with a margin (imported graphs place nodes far out)."""
        m = SCENE_MARGIN
        self.setSceneRect(SCENE.united(self.itemsBoundingRect().adjusted(-m, -m, m, m)))

    def selected_ids(self) -> list[str]:
        return [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]

    def selected_links(self) -> list[LinkItem]:
        return [i for i in self.selectedItems() if isinstance(i, LinkItem)]

    def node_moved_live(self, item: NodeItem) -> None:
        if self._syncing:
            return
        for link in self.links:
            if link.src.node is item or link.dst.node is item:
                link.update_path()

    def snap(self, pos: QPointF, item) -> QPointF:
        from PySide6.QtWidgets import QApplication
        if self._syncing or QApplication.keyboardModifiers() & Qt.ControlModifier:
            return pos
        return QPointF(round(pos.x() / GRID) * GRID, round(pos.y() / GRID) * GRID)

    # --- background ---------------------------------------------------------------------------

    def drawBackground(self, p, rect: QRectF):
        """A dot every two grid steps, hidden when they would crowd."""
        p.fillRect(rect, QColor("#2e2b28"))
        scale = p.transform().m11()
        step = GRID * 2
        if step * scale < 8:
            return
        p.setPen(QPen(QColor("#45403b"), 2.2 / scale))
        x0 = int(rect.left() // step) * step
        y = int(rect.top() // step) * step
        while y < rect.bottom():
            x = x0
            while x < rect.right():
                p.drawPoint(QPointF(x, y))
                x += step
            y += step

    # --- connections --------------------------------------------------------------------------

    def _port_at(self, pos: QPointF) -> PortItem | None:
        for item in self.items(pos):
            if isinstance(item, PortItem):
                return item
        return None

    def _link_at(self, pos: QPointF) -> LinkItem | None:
        return next((i for i in self.items(pos) if isinstance(i, LinkItem)), None)

    def _show_dot_preview(self, pos: QPointF | None) -> None:
        """The Dot an Alt+click would put on the link under the mouse (None: hidden)."""
        if pos is None:
            if self._dot_preview is not None:
                self._dot_preview.hide()
            return
        if self._dot_preview is None:
            self._dot_preview = QGraphicsEllipseItem(-DOT / 2, -DOT / 2, DOT, DOT)
            self._dot_preview.setPen(QPen(QColor("#f0f0f0"), 1.5, Qt.DashLine))
            self._dot_preview.setZValue(6)
            self._dot_preview.setAcceptedMouseButtons(Qt.NoButton)
            self.addItem(self._dot_preview)
        self._dot_preview.setPos(pos)
        self._dot_preview.show()

    def mousePressEvent(self, e):
        port = self._port_at(e.scenePos()) if e.button() == Qt.LeftButton else None
        if port is None:
            self._pending = None
            link = self._link_at(e.scenePos()) if e.button() == Qt.LeftButton else None
            if link is not None and e.modifiers() & Qt.AltModifier and hasattr(self.ctl, "insert_dot"):
                self._show_dot_preview(None)
                self.ctl.insert_dot(link.dst.node.node_id, link.dst.name, e.scenePos())
                e.accept()
                return
            return super().mousePressEvent(e)
        mods = e.modifiers()
        if mods & Qt.AltModifier:  # remove every link of this connector
            self.ctl.disconnect_port(port.node.node_id, port.name, port.is_input)
            e.accept()
            return
        moved = []
        if mods & (Qt.ShiftModifier | Qt.ControlModifier):
            moved = [l for l in self.links if (l.dst if port.is_input else l.src) is port]
        line = QGraphicsPathItem()
        line.setPen(QPen(port_color(port.type), 2, Qt.DashLine))
        line.setZValue(5)
        self.addItem(line)
        mode = "move" if mods & Qt.ShiftModifier else "copy" if mods & Qt.ControlModifier else "new"
        self._drag = (port, line, mode, moved, e.scenePos())
        e.accept()

    def mouseMoveEvent(self, e):
        if self._drag is None:
            alt = e.modifiers() & Qt.AltModifier and not e.buttons() and hasattr(self.ctl, "insert_dot")
            self._show_dot_preview(e.scenePos() if alt and self._link_at(e.scenePos()) is not None else None)
            return super().mouseMoveEvent(e)
        self.drag_link_to(e.scenePos())

    def dragging_link(self) -> bool:
        return self._drag is not None

    def drag_link_to(self, b) -> None:
        """The link being dragged ends at scene point b (the mouse, also while the view scrolls under it)."""
        port, line, *_ = self._drag
        a = port.scenePos()
        line.setPath(link_path(b, a, self.rectangular) if port.is_input else link_path(a, b, self.rectangular))

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            super().mouseReleaseEvent(e)
            moved = {i.node_id: (i.pos().x(), i.pos().y()) for i in self.selectedItems() if isinstance(i, NodeItem)}
            if moved and e.button() == Qt.LeftButton:
                self.ctl.move_nodes(moved)
            return
        port, line, mode, moved, start = self._drag
        self._drag = None
        self.removeItem(line)
        target = self._port_at(e.scenePos())
        if target is port or (target is None and (e.scenePos() - start).manhattanLength() < 4):
            # A click on a connector: the first half of a click-click connection.
            if self._pending is not None and self._pending is not port:
                self._connect(self._pending, port)
                self._pending = None
            else:
                self._pending = port
            return
        self._pending = None
        if target is None:
            if mode == "move" and moved:
                self.ctl.remove_links([(l.dst.node.node_id, l.dst.name) for l in moved])
            elif mode == "new" and not port.is_input and self.ctl.drag_connector_menu:
                self.ctl.open_node_menu(e.scenePos(), (port.node.node_id, port.name))
            return
        if mode in ("move", "copy") and moved:
            self.ctl.relink(moved, port, target, keep=(mode == "copy"))
        else:
            self._connect(port, target)

    def _connect(self, a: PortItem, b: PortItem) -> None:
        if a.is_input == b.is_input or a.node is b.node:
            return
        out, inp = (b, a) if a.is_input else (a, b)
        self.ctl.connect(out.node.node_id, out.name, inp.node.node_id, inp.name)

    def mouseDoubleClickEvent(self, e):
        for item in self.items(e.scenePos()):
            if isinstance(item, NodeItem):
                self.ctl.view_2d_node(item.node_id)
                return
        super().mouseDoubleClickEvent(e)

    # --- menus and keys -----------------------------------------------------------------------

    def contextMenuEvent(self, e):
        node = next((i for i in self.items(e.scenePos()) if isinstance(i, NodeItem)), None)
        menu = QMenu()
        self.ctl.fill_context_menu(menu, node.node_id if node else None, e.scenePos())
        menu.exec(e.screenPos())

    def keyPressEvent(self, e):
        key, mods = e.key(), e.modifiers()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            nodes, links = self.selected_ids(), self.selected_links()
            items = [i.item_id for i in self.selectedItems() if isinstance(i, GraphItem)]
            if items and hasattr(self.ctl, "remove_items"):
                self.ctl.remove_items(items)
            if nodes or links:
                self.ctl.delete(nodes, [(l.dst.node.node_id, l.dst.name) for l in links],
                                relink=(key == Qt.Key_Backspace))
            return
        if mods & (Qt.ControlModifier | Qt.MetaModifier) and key == Qt.Key_A:  # Cmd or Control on macOS
            for item in self.nodes.values():
                item.setSelected(True)
            return
        super().keyPressEvent(e)

    def item_changed(self, item: GraphItem, moved_nodes: dict, resized: bool, moved_items: dict) -> None:
        """A frame/comment/pin was moved (with the nodes and items a frame carries) or resized."""
        self.ctl.update_item(item.item_id, [item.pos().x(), item.pos().y()], item.data.get("size"), moved_nodes,
                             moved_items)

    def select_stream(self, node_id: str, upstream: bool) -> None:
        graph = self.ctl.graph
        ids = graph.upstream([node_id]) if upstream else graph.downstream(node_id)
        for k in ids:
            if k in self.nodes:
                self.nodes[k].setSelected(True)

    def set_thumbnail(self, node_id: str, image, info: str, key=None) -> None:
        item = self.nodes.get(node_id)
        if item is not None:
            item.thumbnail, item.info, item.thumbnail_key = image, info, key
            item.update()
