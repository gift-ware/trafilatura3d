"""The function graph editor, opened inside the Graph panel (with a breadcrumb back
to the graph): function nodes with typed connectors colored by value type, the same gestures as the
Graph view, Space for the function node menu, right click to add nodes and set the output node.

The function edited lives in a node of the package: a FUNCTION parameter (Pixel processor's Per
pixel function…) or a parameter function (node.functions). Every change is a document edit."""

from __future__ import annotations

import copy

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QPushButton, QVBoxLayout, QWidget)

from trafilatura3d.editor.graph.items import LinkItem, NodeItem
from trafilatura3d.editor.graph.nodemenu import fuzzy_score
from trafilatura3d.editor.graph.scene import GraphScene
from trafilatura3d.editor.graph.view import GraphView
from trafilatura3d.functions import FNODES, FunctionError, FunctionGraph, PIXEL_BUILTINS

CATEGORY_COLOR = {"Constant": "#5a5a8a", "Vector": "#4f7a4f", "Swizzle": "#4f7a4f", "Variables": "#8a6a2a",
                  "Samplers": "#2f6f8f", "Cast": "#6a6a6a", "Operator": "#7a4c4c", "Logical": "#7a7a2a",
                  "Comparison": "#7a7a2a", "Function": "#7a4c9a", "Control": "#9b1d1d"}


class FunctionScene(GraphScene):
    def sync_function(self, data: dict, types: dict) -> None:
        """Items for the function graph `data`; `types`: node id -> inferred value type (when known)."""
        self._syncing = True
        selected = self.selected_ids()
        nodes = {n["id"]: n for n in data.get("nodes", [])}
        for k in list(self.nodes):
            if k not in nodes:
                self.removeItem(self.nodes.pop(k))
        for k, n in nodes.items():
            f = FNODES.get(n["type"])
            label = function_label(n)
            item = self.nodes.get(k)
            ins = list(f.inputs) if f else []
            if item is None or item.label != label or list(item.inputs) != [p for p, _ in ins]:
                if item is not None:
                    self.removeItem(item)
                item = NodeItem(k, label, n["type"], "", ins, [("output", "")])
                item.color = QColor(CATEGORY_COLOR.get(f.category if f else "", "#555"))
                item.body = 24 + max(len(ins) - 1, 0) * 31  # compact: function nodes have no thumbnail
                item.height = 14 + 2 + item.body + 2
                self.nodes[k] = item
                self.addItem(item)
            item.setPos(*n.get("position", [0, 0]))
            item.setSelected(k in selected)
            item.info = ("★ output  " if data.get("output") == k else "") + types.get(k, "")
            item.outputs["output"].type = types.get(k)
            for port, p in item.inputs.items():
                src = n.get("inputs", {}).get(port)
                p.connected = src in nodes
                p.type = types.get(src) if src else None
        for link in self.links:
            self.removeItem(link)
        self.links = []
        for k, n in nodes.items():
            for port, src in n.get("inputs", {}).items():
                if src in self.nodes and port in self.nodes[k].inputs:
                    self.nodes[src].outputs["output"].connected = True
                    link = LinkItem(self.nodes[src].outputs["output"], self.nodes[k].inputs[port])
                    self.addItem(link)
                    self.links.append(link)
        self._syncing = False
        self.update()


def function_label(n: dict) -> str:
    f = FNODES.get(n["type"])
    if f is None:
        return n["type"]
    p = n.get("params", {})
    if n["type"].startswith("const_") and "value" in p:
        v = p["value"]
        if isinstance(v, list):
            v = ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v)
        return f"{f.label}: {v:.3g}" if isinstance(v, float) else f"{f.label}: {v}"
    if "variable" in p and p["variable"]:
        return f"{f.label.split(' ')[0]} {p['variable']}"
    return f.label


class FunctionNodeMenu(QFrame):
    chosen = Signal(str)

    def __init__(self, parent):
        super().__init__(parent, Qt.Popup)
        self.setMinimumSize(280, 320)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search function nodes…")
        self.list = QListWidget()
        lay.addWidget(self.search)
        lay.addWidget(self.list)
        self.search.textChanged.connect(self.refresh)
        self.list.itemActivated.connect(lambda it: (self.chosen.emit(it.data(Qt.UserRole)), self.close()))
        self.search.returnPressed.connect(self._enter)
        self.refresh("")
        self.search.setFocus()

    def refresh(self, text):
        rows = []
        for t, f in FNODES.items():
            s = fuzzy_score(text, f.label)
            if s is None:
                s = fuzzy_score(text, f.category + " " + f.label)
                if s is None:
                    continue
            rows.append((-s, f.label, f.category, t))
        rows.sort()
        self.list.clear()
        for _, label, cat, t in rows:
            it = QListWidgetItem(f"{label}    —  {cat}")
            it.setData(Qt.UserRole, t)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _enter(self):
        if self.list.currentItem():
            self.chosen.emit(self.list.currentItem().data(Qt.UserRole))
            self.close()


class NodeFunction:
    """A function graph in a node of the package: a FUNCTION parameter (Pixel processor's Per pixel
    function…) or the function driving a parameter (node.functions)."""

    def __init__(self, package, graph_id, node_id, param, is_param_function):
        self.package, self.graph_id, self.node_id = package, graph_id, node_id
        self.param, self.is_param_function = param, is_param_function

    def _graph(self):
        return self.package.graphs[self.graph_id]

    def exists(self) -> bool:
        return self.graph_id in self.package.graphs and self.node_id in self._graph().nodes

    def title(self) -> str:
        d = self._graph().definition(self.node_id)
        kind = "function of " if self.is_param_function else ""
        return f"{self.graph_id}  ›  {d.label} ({self.node_id})  ›  {kind}{d.params[self.param].label}"

    def get(self) -> dict:
        node = self._graph().nodes[self.node_id]
        if self.is_param_function:
            return node.functions.get(self.param, {"nodes": [], "output": None})
        return node.params.get(self.param, self._graph().definition(self.node_id).params[self.param].default)

    def put(self, data: dict) -> None:
        if self.is_param_function:
            self._graph().nodes[self.node_id].functions[self.param] = data
        else:
            self._graph().set_param(self.node_id, self.param, data)

    def variables(self) -> dict:
        from trafilatura3d.functions import BUILTINS, PARAM_TYPES
        g = self._graph()
        if self.is_param_function:
            return g.function_variables()
        builtins = PIXEL_BUILTINS if self._graph().nodes[self.node_id].type == "pixel_processor" else BUILTINS
        values = {g.identifier(n): "float4" for n in g.nodes.values() if n.type == "input_value"}
        return {**builtins, **values, **{pid: PARAM_TYPES[p.param.kind] for pid, p in g.parameters.items()
                                         if p.param.kind in PARAM_TYPES}}


class FxFunction(NodeFunction):
    """The function driving a parameter of a node of an FX-Map graph."""

    def __init__(self, package, graph_id, node_id, fx_node_id, param, fx_param="fxmap"):
        super().__init__(package, graph_id, node_id, param, True)
        self.fx_node_id, self.fx_param = fx_node_id, fx_param

    def _fx(self) -> dict:
        node = self._graph().nodes[self.node_id]
        return node.params.get(self.fx_param, self._graph().definition(self.node_id).params[self.fx_param].default)

    def exists(self) -> bool:
        return super().exists() and any(n["id"] == self.fx_node_id for n in self._fx()["nodes"])

    def title(self) -> str:
        from trafilatura3d.fxmap import FX_PARAMS
        n = next(m for m in self._fx()["nodes"] if m["id"] == self.fx_node_id)
        return (f"{self.graph_id}  ›  Stamp Map ({self.node_id})  ›  {self.fx_node_id}  ›  "
                f"function of {FX_PARAMS[n['type']][self.param].label}")

    def get(self) -> dict:
        n = next(m for m in self._fx()["nodes"] if m["id"] == self.fx_node_id)
        return n.get("functions", {}).get(self.param, {"nodes": [], "output": None})

    def put(self, data: dict) -> None:
        fx = copy.deepcopy(self._fx())
        n = next(m for m in fx["nodes"] if m["id"] == self.fx_node_id)
        n.setdefault("functions", {})[self.param] = data
        self._graph().set_param(self.node_id, self.fx_param, fx)

    def variables(self) -> dict:
        from trafilatura3d.functions import PARAM_TYPES
        from trafilatura3d.fxmap import FX_VARIABLES
        g = self._graph()
        return {**FX_VARIABLES, **{pid: PARAM_TYPES[p.param.kind] for pid, p in g.parameters.items()
                                   if p.param.kind in PARAM_TYPES}}


class FunctionPanel(QWidget):
    """Edits one function graph (see NodeFunction, FxFunction); `back` returns to where it came from."""
    back = Signal()
    selection_changed = Signal(object)  # (panel, function node id) or None

    def __init__(self, document, status):
        super().__init__()
        self.doc, self.status = document, status
        self.package = self.graph_id = self.node_id = None
        self.target = None
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
        holder = QWidget()
        holder.setLayout(crumb)
        lay.addWidget(holder)
        self.scene = FunctionScene(self)
        self.view = GraphView(self.scene)
        lay.addWidget(self.view, 1)
        self.scene.selectionChanged.connect(self._selection)
        self.doc.changed.connect(lambda pkg: self.sync() if pkg is self.package and self.isVisible() else None)
        self._arranging = False  # this panel's own move: the scene is already arranged
        self.doc.layout_changed.connect(
            lambda pkg: self.sync() if pkg is self.package and self.isVisible() and not self._arranging else None)
        from trafilatura3d.editor.keys import shortcut
        local = Qt.WidgetWithChildrenShortcut  # this view's keys: the main graph's view has its own
        menu = lambda: self.open_node_menu(self.view.cursor_scene_pos)
        shortcut("Space", self.view, menu, local)
        shortcut("Tab", self.view, menu, local)
        shortcut(QKeySequence.Copy, self.view, self.copy, local)
        shortcut(QKeySequence.Paste, self.view, lambda: self.paste(self.view.cursor_scene_pos), local)
        shortcut("Ctrl+D", self.view, self.duplicate, local)

    # --- target -------------------------------------------------------------------------------

    def open(self, target) -> None:
        """Edit the function graph a target points at (NodeFunction, FxFunction)."""
        self.target = target
        self.package, self.graph_id, self.node_id = target.package, target.graph_id, target.node_id
        self.crumb.setText(target.title())
        self.scene.clear()
        self.sync()
        QTimer.singleShot(0, self.view, self.view.focus)

    def data(self) -> dict:
        return self.target.get()

    def variables(self) -> dict:
        return self.target.variables()

    def types(self, data) -> dict:
        """Inferred types of the nodes that can be typed (the graph may be incomplete)."""
        try:
            return FunctionGraph(data, self.variables()).types
        except FunctionError as e:
            self.status(f"Function: {e}")
            return FunctionGraph.infer_types(data, self.variables())

    def sync(self):
        if self.target is None or not self.target.exists():
            if self.target is not None:
                self.back.emit()
            return
        data = self.data()
        self.scene.sync_function(data, self.types(data))

    def edit(self, label, mutate, merge=None, layout=False):
        """Apply `mutate(copy)` to a copy of the function graph and store it; a `layout` edit is a move made in this
        view (nothing recooks, the scene already shows it)."""
        new = copy.deepcopy(self.data())

        def do():
            mutate(new)
            self.target.put(new)  # typed when compiled: editing passes through incomplete graphs
        self._arranging = layout
        try:
            self.doc.edit(self.package, label, do, merge, layout)
        finally:
            self._arranging = False

    # --- controller: called by the scene --------------------------------------------------------

    @property
    def graph(self):
        return _FunctionTopology(self.data())

    def node_label(self, n, d):
        return function_label(n)

    def connect(self, src, out, dst, port):
        self.edit("Connect", lambda f: _node(f, dst).setdefault("inputs", {}).__setitem__(port, src))

    def disconnect_port(self, node_id, port, is_input):
        def do(f):
            if is_input:
                _node(f, node_id).get("inputs", {}).pop(port, None)
            else:
                for n in f["nodes"]:
                    for p in [p for p, s in n.get("inputs", {}).items() if s == node_id]:
                        del n["inputs"][p]
        self.edit("Disconnect", do)

    def remove_links(self, links):
        self.edit("Remove links", lambda f: [_node(f, d).get("inputs", {}).pop(p, None) for d, p in links])

    def relink(self, links, anchor, target, keep):
        def do(f):
            for link in links:
                src, dst, port = link.src.node.node_id, link.dst.node.node_id, link.dst.name
                if not keep:
                    _node(f, dst).get("inputs", {}).pop(port, None)
                if anchor.is_input:
                    _node(f, target.node.node_id).setdefault("inputs", {})[target.name] = src
                else:
                    _node(f, dst).setdefault("inputs", {})[port] = target.node.node_id
        self.edit("Move links", do)

    def move_nodes(self, positions):
        def do(f):
            for k, (x, y) in positions.items():
                _node(f, k)["position"] = [x, y]
        self.edit("Move nodes", do, layout=True)

    def add_node(self, node_type, pos: QPointF, connect_from=None):
        new_id = []

        def do(f):
            n = 1
            ids = {m["id"] for m in f["nodes"]}
            while f"{node_type}{n}" in ids:
                n += 1
            k = f"{node_type}{n}"
            f["nodes"].append({"id": k, "type": node_type, "params": {}, "inputs": {}, "position": [pos.x(), pos.y()]})
            if connect_from and FNODES[node_type].inputs:
                f["nodes"][-1]["inputs"][FNODES[node_type].inputs[0][0]] = connect_from[0]
            if not f.get("output"):
                f["output"] = k
            new_id.append(k)
        self.edit(f"Add {node_type}", do)
        return new_id[0] if new_id else None

    def delete(self, nodes, links, relink=False):
        def do(f):
            for d, p in links:
                _node(f, d).get("inputs", {}).pop(p, None)
            f["nodes"] = [n for n in f["nodes"] if n["id"] not in nodes]
            for n in f["nodes"]:
                for p in [p for p, s in n.get("inputs", {}).items() if s in nodes]:
                    del n["inputs"][p]
            if f.get("output") in nodes:
                f["output"] = f["nodes"][-1]["id"] if f["nodes"] else None
        self.edit("Delete", do)

    def set_output(self, node_id):
        self.edit("Set as output", lambda f: f.__setitem__("output", node_id))

    def copy(self):
        ids = set(self.scene.selected_ids())
        self.clipboard = [copy.deepcopy(n) for n in self.data()["nodes"] if n["id"] in ids] or None

    def paste(self, pos):
        if not self.clipboard:
            return
        x0 = min(n.get("position", [0, 0])[0] for n in self.clipboard)
        y0 = min(n.get("position", [0, 0])[1] for n in self.clipboard)

        def do(f):
            ids = {m["id"] for m in f["nodes"]}
            mapping = {}
            for n in self.clipboard:
                k, i = n["id"], 1
                while k in ids:
                    i += 1
                    k = f"{n['type']}{i}"
                ids.add(k)
                mapping[n["id"]] = k
            for n in self.clipboard:
                m = copy.deepcopy(n)
                m["id"] = mapping[n["id"]]
                m["inputs"] = {p: mapping[s] for p, s in n.get("inputs", {}).items() if s in mapping}
                px, py = n.get("position", [0, 0])
                m["position"] = [pos.x() + px - x0, pos.y() + py - y0]
                f["nodes"].append(m)
        self.edit("Paste", do)

    def duplicate(self):
        self.copy()
        if self.clipboard:
            p = self.clipboard[0].get("position", [0, 0])
            self.paste(QPointF(p[0] + 32, p[1] + 32))

    def open_node_menu(self, scene_pos, from_port=None):
        menu = FunctionNodeMenu(self)
        menu.chosen.connect(lambda t: self.add_node(t, scene_pos, from_port))
        menu.move(self.view.mapToGlobal(self.view.mapFromScene(scene_pos)))
        menu.show()

    def view_2d_node(self, node_id):
        self.set_output(node_id)  # double click: a "Set as output node" shortcut

    def fill_context_menu(self, menu: QMenu, node_id, pos):
        if node_id is not None:
            menu.addAction("Set as output node", lambda: self.set_output(node_id))
            menu.addAction("Delete", lambda: self.delete([node_id], []))
            menu.addAction("Copy", self.copy)
            menu.addAction("Duplicate", self.duplicate)
            return
        add = menu.addMenu("Add node")
        cats = {}
        for t, f in FNODES.items():
            cats.setdefault(f.category, []).append((f.label, t))
        for cat in ["Constant", "Vector", "Swizzle", "Variables", "Samplers", "Cast", "Operator", "Logical",
                    "Comparison", "Function", "Control"]:
            sub = add.addMenu(cat)
            for label, t in sorted(cats.get(cat, [])):
                sub.addAction(label, lambda t=t: self.add_node(t, pos))
        paste = menu.addAction("Paste", lambda: self.paste(pos))
        paste.setEnabled(bool(self.clipboard))

    def _selection(self):
        ids = self.scene.selected_ids()
        self.selection_changed.emit((self, ids[0]) if len(ids) == 1 else None)

    # Properties edits a function node's parameters through this.
    def set_function_param(self, node_id, name, value, merge=None):
        self.edit(f"Edit {name}", lambda f: _node(f, node_id).setdefault("params", {}).__setitem__(name, value), merge)


class _FunctionTopology:
    """upstream/downstream over a function graph (for the scene's select upstream/downstream)."""

    def __init__(self, data):
        self.nodes = {n["id"]: n for n in data.get("nodes", [])}

    def upstream(self, ids):
        out, stack = [], list(ids)
        while stack:
            k = stack.pop()
            if k in self.nodes and k not in out:
                out.append(k)
                stack.extend(self.nodes[k].get("inputs", {}).values())
        return out

    def downstream(self, k):
        out, stack = set(), [k]
        while stack:
            c = stack.pop()
            if c not in out:
                out.add(c)
                stack.extend(n["id"] for n in self.nodes.values() if c in n.get("inputs", {}).values())
        return out


def _node(f, k):
    return next(n for n in f["nodes"] if n["id"] == k)
