"""The Stamp Map graph editor (the FX-Map engine), opened inside the Graph panel from a Stamp Map node: Quadrant,
Iterate and Switch nodes; a node's outputs are its child branches (Quadrant 1-4, Iterate Pass-through/Repeated,
Switch Input 1/2), its input the link to its parent; the root is marked ★. The tree flows left to
right. Every change is a document edit of the FX-Map node."""

from __future__ import annotations

import copy

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from trafilatura3d.editor.graph.function_panel import _FunctionTopology
from trafilatura3d.editor.graph.items import LinkItem, NodeItem
from trafilatura3d.editor.graph.scene import GraphScene
from trafilatura3d.editor.graph.view import GraphView
from trafilatura3d.fxmap import FX_PARAMS, SLOT_LABELS, SLOTS

COLORS = {"quadrant": "#c47a1a", "iterate": "#3a6ea5", "switch": "#7a4c9a"}
LABELS = {"quadrant": "Quadrant", "iterate": "Iterate", "switch": "Switch"}


class FxScene(GraphScene):
    def sync_fx(self, data: dict) -> None:
        self._syncing = True
        selected = self.selected_ids()
        nodes = {n["id"]: n for n in data.get("nodes", [])}
        for k in list(self.nodes):
            if k not in nodes:
                self.removeItem(self.nodes.pop(k))
        for k, n in nodes.items():
            kind = n["type"]
            label = fx_label(n)
            item = self.nodes.get(k)
            if item is None or item.label != label:
                if item is not None:
                    self.removeItem(item)
                outs = [(s, SLOT_LABELS[s]) for s in SLOTS.get(kind, ())]
                item = NodeItem(k, label, kind, "", [("parent", "Parent")], outs)
                item.color = QColor(COLORS.get(kind, "#555"))
                item.body = 24 + max(len(outs) - 1, 0) * 31
                item.height = 14 + 2 + item.body + 2
                item.show_names = True
                self.nodes[k] = item
                self.addItem(item)
            item.setPos(*n.get("position", [0, 0]))
            item.setSelected(k in selected)
            item.info = "★ root" if data.get("root") == k else ""
            for port in item.outputs.values():
                port.connected = port.name in n.get("children", {})
            item.inputs["parent"].connected = any(k in m.get("children", {}).values() for m in nodes.values())
        for link in self.links:
            self.removeItem(link)
        self.links = []
        for k, n in nodes.items():
            for slot, child in n.get("children", {}).items():
                if child in self.nodes and slot in self.nodes[k].outputs:
                    link = LinkItem(self.nodes[k].outputs[slot], self.nodes[child].inputs["parent"])
                    self.addItem(link)
                    self.links.append(link)
        self._syncing = False
        self.update()


def fx_label(n: dict) -> str:
    if n["type"] == "quadrant":
        p = n.get("params", {}).get("pattern", "No Pattern")
        return "Quadrant" if p == "No Pattern" else f"Quadrant: {p}"
    if n["type"] == "iterate":
        return f"Iterate ×{n.get('params', {}).get('iterations', FX_PARAMS['iterate']['iterations'].default)}"
    return LABELS.get(n["type"], n["type"])


class FxMapPanel(QWidget):
    back = Signal()
    selection_changed = Signal(object)  # (panel, FX-Map node id) or None
    open_function = Signal(object)  # an FxFunction target

    def __init__(self, document, status):
        super().__init__()
        self.doc, self.status = document, status
        self.package = self.graph_id = self.node_id = None
        self.param = "fxmap"
        self.clipboard = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        crumb = QHBoxLayout()
        crumb.setContentsMargins(6, 3, 6, 3)
        b = QPushButton("◀ Back")
        b.clicked.connect(self.back)
        crumb.addWidget(b)
        self.crumb = QLabel()
        f = QFont()
        f.setBold(True)
        self.crumb.setFont(f)
        crumb.addWidget(self.crumb, 1)
        for kind in ("quadrant", "iterate", "switch"):
            add = QPushButton(f"+ {LABELS[kind]}")
            add.clicked.connect(lambda _=False, k=kind: self.add_node(k, self.view.mapToScene(
                self.view.viewport().rect().center())))
            crumb.addWidget(add)
        holder = QWidget()
        holder.setLayout(crumb)
        lay.addWidget(holder)
        self.scene = FxScene(self)
        self.view = GraphView(self.scene)
        lay.addWidget(self.view, 1)
        self.scene.selectionChanged.connect(self._selection)
        self.doc.changed.connect(lambda pkg: self.sync() if pkg is self.package and self.isVisible() else None)
        self._arranging = False  # this panel's own move: the scene is already arranged
        self.doc.layout_changed.connect(
            lambda pkg: self.sync() if pkg is self.package and self.isVisible() and not self._arranging else None)

    def open(self, package, graph_id, node_id, param="fxmap"):
        self.package, self.graph_id, self.node_id, self.param = package, graph_id, node_id, param
        self.crumb.setText(f"{graph_id}  ›  Stamp Map ({node_id})")
        self.scene.clear()
        self.sync()
        QTimer.singleShot(0, self.view.focus)

    def data(self) -> dict:
        g = self.package.graphs[self.graph_id]
        return g.nodes[self.node_id].params.get(self.param, g.definition(self.node_id).params[self.param].default)

    def sync(self):
        if self.package is None:
            return
        if self.graph_id not in self.package.graphs or self.node_id not in self.package.graphs[self.graph_id].nodes:
            self.back.emit()
            return
        self.scene.sync_fx(self.data())

    def edit(self, label, mutate, merge=None, layout=False):
        """Apply `mutate(copy)` to a copy of the FX-Map graph and store it; a `layout` edit is a move made in this
        view (nothing recooks, the scene already shows it)."""
        g = self.package.graphs[self.graph_id]
        new = copy.deepcopy(self.data())

        def do():
            mutate(new)
            g.set_param(self.node_id, self.param, new)
        self._arranging = layout
        try:
            self.doc.edit(self.package, label, do, merge, layout)
        finally:
            self._arranging = False

    # --- controller: called by the scene --------------------------------------------------------

    @property
    def graph(self):
        data = self.data()
        return _FunctionTopology({"nodes": [{"id": n["id"], "inputs": n.get("children", {})}
                                            for n in data["nodes"]]})

    def connect(self, src, out, dst, port):
        """src's child slot `out` -> dst (port is "parent")."""
        def do(f):
            _node(f, src).setdefault("children", {})[out] = dst
        self.edit("Connect", do)

    def disconnect_port(self, node_id, port, is_input):
        def do(f):
            if is_input:  # detach from every parent
                for n in f["nodes"]:
                    for slot in [s for s, c in n.get("children", {}).items() if c == node_id]:
                        del n["children"][slot]
            else:
                _node(f, node_id).get("children", {}).pop(port, None)
        self.edit("Disconnect", do)

    def remove_links(self, links):
        """links: (child id, "parent") pairs from the scene; remove the slots pointing at them."""
        self.edit("Remove links", lambda f: [self._unlink(f, child) for child, _ in links])

    def _unlink(self, f, child):
        for n in f["nodes"]:
            for slot in [s for s, c in n.get("children", {}).items() if c == child]:
                del n["children"][slot]

    def relink(self, links, anchor, target, keep):
        def do(f):
            for link in links:
                parent, slot, child = link.src.node.node_id, link.src.name, link.dst.node.node_id
                if not keep:
                    _node(f, parent)["children"].pop(slot, None)
                if anchor.is_input:  # the child end moved to another node
                    _node(f, parent).setdefault("children", {})[slot] = target.node.node_id
                else:  # the parent end moved to another slot
                    _node(f, target.node.node_id).setdefault("children", {})[target.name] = child
        self.edit("Move links", do)

    def move_nodes(self, positions):
        def do(f):
            for k, (x, y) in positions.items():
                _node(f, k)["position"] = [x, y]
        self.edit("Move nodes", do, layout=True)

    def add_node(self, kind, pos: QPointF, connect_from=None):
        def do(f):
            n = 1
            ids = {m["id"] for m in f["nodes"]}
            while f"{kind}{n}" in ids:
                n += 1
            k = f"{kind}{n}"
            f["nodes"].append({"id": k, "type": kind, "params": {}, "children": {}, "functions": {},
                               "position": [pos.x(), pos.y()]})
            if connect_from:
                _node(f, connect_from[0]).setdefault("children", {})[connect_from[1]] = k
            if not f.get("root"):
                f["root"] = k
        self.edit(f"Add {kind}", do)

    def delete(self, nodes, links, relink=False):
        def do(f):
            for child, _ in links:
                self._unlink(f, child)
            f["nodes"] = [n for n in f["nodes"] if n["id"] not in nodes]
            for n in f["nodes"]:
                for slot in [s for s, c in n.get("children", {}).items() if c in nodes]:
                    del n["children"][slot]
            if f.get("root") in nodes:
                f["root"] = f["nodes"][0]["id"] if f["nodes"] else None
        self.edit("Delete", do)

    def set_root(self, node_id):
        self.edit("Set as root", lambda f: f.__setitem__("root", node_id))

    def open_node_menu(self, scene_pos, from_port=None):
        menu = QMenu(self)
        for kind in ("quadrant", "iterate", "switch"):
            menu.addAction(LABELS[kind], lambda k=kind: self.add_node(k, scene_pos, from_port))
        menu.exec(self.view.mapToGlobal(self.view.mapFromScene(scene_pos)))

    def view_2d_node(self, node_id):
        self.set_root(node_id)

    def fill_context_menu(self, menu: QMenu, node_id, pos):
        if node_id is not None:
            menu.addAction("Set as root", lambda: self.set_root(node_id))
            menu.addAction("Delete", lambda: self.delete([node_id], []))
            return
        for kind in ("quadrant", "iterate", "switch"):
            menu.addAction(f"Add {LABELS[kind]}", lambda k=kind: self.add_node(k, pos))

    def _selection(self):
        ids = self.scene.selected_ids()
        self.selection_changed.emit((self, ids[0]) if len(ids) == 1 else None)

    # Properties edits FX-Map node parameters and functions through these.
    def set_fx_param(self, fx_id, name, value, merge=None):
        self.edit(f"Edit {name}", lambda f: _node(f, fx_id).setdefault("params", {}).__setitem__(name, value), merge)

    def remove_fx_function(self, fx_id, name):
        self.edit("Remove function", lambda f: _node(f, fx_id).setdefault("functions", {}).pop(name, None))

    def set_fx_function(self, fx_id, name, function):
        self.edit("Set function", lambda f: _node(f, fx_id).setdefault("functions", {}).__setitem__(name, function))


def _node(f, k):
    return next(n for n in f["nodes"] if n["id"] == k)
