"""Graph operations for scripts and remote control, without Qt: a compact description of a graph, node types
described as data, a batch of edits by name (`apply_ops`), the automatic layout, and per-node cook timings.

The node geometry of the Graph view lives here too (`node_size`, `port_y`, the Dot's `DOT`, the links' curve): the
layout spaces nodes the way the view draws them, and the view reads the same numbers."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from trafilatura3d.nodes.base import Node, Param, ParamKind
from trafilatura3d.package import DocumentError, Graph

# Node geometry in scene units (the Graph view's items): the title over a square thumbnail, connectors
# on the sides one PORT_STEP apart.
THUMB = 96
TITLE = 14
PORT_R = 6.0
PORT_STEP = 31.0
MARGIN = 2.0
GRID = 16  # the view's grid: nodes snap to it
COLUMN = 192  # layout: from a node's left side to the next column's
GAP = 32  # layout: between two nodes of a column (the size line under a node fits)


DOT = 16.0  # a Dot node: a disc this wide on the link, its connectors on its sides at mid-height
DOT_TYPE = "dot"
FRAME_BAR = 22  # a frame's title bar, as the Graph view draws it
FRAME_PAD = 24  # a frame's margin around its nodes (the size line under a node fits)
FRAME_SPACE = 32  # layout: between a frame's border and what is next to it in a column (the grid snap moves a
# node up to half a GRID either way: it never reaches the frame)


def node_size(inputs: int, outputs: int, dot: bool = False) -> tuple[float, float]:
    """A node's width and height with this many connectors on its sides (a Dot: its disc)."""
    if dot:
        return DOT, DOT
    n = max(inputs, outputs, 1)
    body = max(THUMB, (n - 1) * PORT_STEP + 2 * PORT_R + 8)
    return THUMB + 2 * MARGIN, TITLE + MARGIN + body + MARGIN


def port_y(index: int, dot: bool = False) -> float:
    """A connector's height from the node's top."""
    if dot:
        return DOT / 2
    return TITLE + MARGIN + PORT_R + 4 + index * PORT_STEP


def link_controls(a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two control points of the cubic curve a link is drawn with, from output connector a to input b."""
    dx = max(abs(b[0] - a[0]) * 0.5, 30.0)
    return (a[0] + dx, a[1]), (b[0] - dx, b[1])


def _node_rect(graph: Graph, node_id: str) -> tuple[float, float, float, float]:
    x, y = graph.nodes[node_id].position
    w, h = _shape(graph, node_id)[:2]
    return x, y, x + w, y + h


# Frames. A frame made around given nodes and frames (the `frame` op) keeps their lists (`nodes`, `frames`); a
# frame drawn by hand holds what lies in it, as when it is dragged: the nodes whose centre is in its box and the
# frames whose box is in it, each in the smallest such frame. A frame in a frame moves with it and is fitted first.

def _frames(graph: Graph) -> list[dict]:
    return [i for i in graph.items if i["kind"] == "frame"]


def _frame_box(f: dict) -> tuple[float, float, float, float]:
    (x, y), (w, h) = f["position"], f["size"]
    return x, y, x + w, y + h


def _inside(r, box) -> bool:
    return box[0] <= r[0] and r[2] <= box[2] and box[1] <= r[1] and r[3] <= box[3]


def _smallest_holder(frames: list[dict], fits) -> str | None:
    holders = [f for f in frames if "nodes" not in f and fits(_frame_box(f))]
    return min(holders, key=lambda f: f["size"][0] * f["size"][1])["id"] if holders else None


def frame_parents(graph: Graph) -> dict[str, str | None]:
    """Each frame's parent frame (None at the top)."""
    frames = _frames(graph)
    parent: dict[str, str | None] = {f["id"]: None for f in frames}
    for f in frames:
        for c in f.get("frames", []):
            if c in parent:
                parent[c] = f["id"]
    for f in frames:
        if parent[f["id"]] is None and "nodes" not in f:
            box = _frame_box(f)
            parent[f["id"]] = _smallest_holder([g for g in frames if g is not f and g["size"][0] * g["size"][1]
                                                > f["size"][0] * f["size"][1]], lambda b: _inside(box, b))
    return parent


def frame_members(graph: Graph) -> dict[str, list[str]]:
    """The nodes directly in each frame (not those of the frames in it)."""
    frames = _frames(graph)
    out: dict[str, list[str]] = {f["id"]: [k for k in f.get("nodes", []) if k in graph.nodes] for f in frames}
    listed = {k for f in frames for k in f.get("nodes", [])}
    for k in graph.nodes:
        if k not in listed:
            x0, y0, x1, y1 = _node_rect(graph, k)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            holder = _smallest_holder(frames, lambda b: b[0] <= cx <= b[2] and b[1] <= cy <= b[3])
            if holder:
                out[holder].append(k)
    return out


def _frame_paths(graph: Graph) -> tuple[dict, dict, dict]:
    """(parent of each frame, nodes directly in each frame, each node's frames from the outermost in)."""
    parent, members = frame_parents(graph), frame_members(graph)

    def chain(fid):
        out = []
        while fid is not None and fid not in out:
            out.append(fid)
            fid = parent[fid]
        return out[::-1]
    path = {k: chain(fid) for fid, ks in members.items() for k in ks}
    return parent, members, path


def fit_frames(graph: Graph, parent: dict, members: dict, only: set | None = None) -> None:
    """Fit frames (`only` these, else all) around their nodes and the frames in them, the innermost first, with the
    membership given (taken before the nodes moved)."""
    depth = {}
    for fid in parent:
        d, p = 0, parent[fid]
        while p is not None and d <= len(parent):
            d, p = d + 1, parent[p]
        depth[fid] = d
    children: dict[str, list[str]] = {fid: [] for fid in parent}
    for fid, p in parent.items():
        if p is not None:
            children[p].append(fid)
    for fid in sorted(parent, key=lambda f: -depth[f]):
        if only is not None and fid not in only:
            continue
        rects = [_node_rect(graph, k) for k in members[fid] if k in graph.nodes] + \
                [_frame_box(graph.item(c)) for c in children[fid]]
        if rects:
            fit_frame(graph.item(fid), rects)


def fit_frame(frame: dict, rects: list[tuple[float, float, float, float]]) -> None:
    """Place and size a frame around these boxes (its title bar above them)."""
    x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
    x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
    frame["position"] = [x0 - FRAME_PAD, y0 - FRAME_PAD - FRAME_BAR]
    frame["size"] = [x1 - x0 + 2 * FRAME_PAD, y1 - y0 + 2 * FRAME_PAD + FRAME_BAR]


def _ports(graph: Graph, node_id: str) -> tuple[list[str], list[str]]:
    """The connectors the view shows: (inputs, outputs); a node whose definition is missing shows its links."""
    node = graph.nodes[node_id]
    try:
        d = graph.definition(node_id)
    except (DocumentError, OSError, ValueError):
        outs = list(node.outputs) or sorted({o for m in graph.nodes.values() for s, o in m.inputs.values()
                                             if s == node_id}) or ["output"]
        return list(node.inputs), outs
    values = d.values(node)
    shown_in, shown_out = set(d.visible_inputs(values)), set(d.visible_outputs(values))
    return [k for k in d.inputs if k in shown_in], [k for k in d.outputs if k in shown_out]


# --- description ------------------------------------------------------------------------------------

def _value(param: Param, value: Any) -> Any:
    """A parameter value as shown in a description: function and FX-Map graphs summarized."""
    if param.kind in (ParamKind.FUNCTION, ParamKind.FXMAP):
        nodes = value.get("nodes", []) if isinstance(value, dict) else []
        return f"<{param.kind.value} graph: {len(nodes)} nodes>"
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list) and all(isinstance(x, float) for x in value):
        return [round(x, 4) for x in value]
    return value


def describe(graph: Graph) -> dict:
    """A compact description: nodes (type, position, parameters that differ from the default), links
    `src.output -> dst.port`, the graph's inputs and its outputs with their usage."""
    nodes, links, outputs = {}, [], []
    for n in graph.nodes.values():
        entry: dict[str, Any] = {"type": n.type, "pos": [round(n.position[0]), round(n.position[1])]}
        if n.ref:
            entry["ref"] = n.ref
        try:
            d = graph.definition(n.id)
        except (DocumentError, OSError, ValueError) as e:
            d = None
            entry["missing"] = str(e)
        if d is not None:
            params = {k: _value(d.params[k], v) for k, v in n.params.items()
                      if k in d.params and v != d.params[k].default}
            if params:
                entry["params"] = params
        if n.base:
            entry["base"] = n.base
        if n.links:
            entry["linked"] = n.links  # parameter -> graph input driving it
        if n.functions:
            entry["functions"] = sorted(n.functions)  # parameters driven by a function graph
        if n.disabled:
            entry["disabled"] = True
        nodes[n.id] = entry
        for port, (src, out) in n.inputs.items():
            links.append(f"{src}.{out} -> {n.id}.{port}")
        if n.type == "output" and d is not None:
            v = d.values(n)
            outputs.append({"node": n.id, "identifier": v["identifier"], "usage": v["usage"]})
    out = {"graph": graph.id, "parent_size": [1 << graph.parent_size[0], 1 << graph.parent_size[1]],
           "nodes": nodes, "links": links, "outputs": outputs}
    if graph.base:
        out["base"] = graph.base
    if graph.parameters:
        out["inputs"] = {pid: _param_info(ep.param) for pid, ep in graph.parameters.items()}
    if graph.items:
        parent, members, _ = _frame_paths(graph)
        out["items"] = []
        for i in graph.items:
            entry = {"id": i["id"], "kind": i["kind"], **({"title": i["title"]} if "title" in i else {}),
                     **({"text": i["text"]} if i.get("text") else {})}
            if i["kind"] == "frame":
                entry["nodes"] = members[i["id"]]
                entry["frames"] = [f for f, p in parent.items() if p == i["id"]]
                if parent[i["id"]]:
                    entry["in"] = parent[i["id"]]
            out["items"].append(entry)
    warnings = graph_warnings(graph)
    if warnings:
        out["warnings"] = warnings
    return out


def graph_warnings(graph: Graph) -> list[str]:
    """The graph's warnings as the Graph view lists them (diagnostics.graph_warnings), as text."""
    from trafilatura3d.diagnostics import graph_warnings as found
    return [(f"{w.node}: " if w.node else "") + w.message + (f" ({w.detail})" if w.detail else "")
            for w in found(graph.package, graph.id)]


def _param_info(p: Param) -> dict:
    info: dict[str, Any] = {"kind": p.kind.value, "default": _value(p, p.default)}
    if p.editor:
        info["editor"] = p.editor_name()
    if p.label:
        info["label"] = p.label
    if p.min is not None or p.max is not None:
        info["range"] = [p.min, p.max]
    if p.choices:
        info["choices"] = list(p.choices)
    if p.items:
        info["items"] = [list(i) for i in p.items]
    return info


def node_info(d: Node) -> dict:
    """A node type as data: its ports and parameters (kind, default, range, choices)."""
    info: dict[str, Any] = {"type": d.type, "label": d.label, "category": d.category,
                            "inputs": {k: i.type.value for k, i in d.inputs.items()},
                            "outputs": {k: o.type.value for k, o in d.outputs.items()},
                            "params": {k: _param_info(p) for k, p in d.params.items()}}
    if d.counted_inputs:
        info["counted_inputs"] = d.counted_inputs  # count parameter -> numbered inputs it enables
    info["inherits"] = inheritance(d)
    info["base"] = base_info()
    return info


def inheritance(d: Node) -> dict:
    """Where a node of this type takes its size and format from, as the compiler resolves it: the default mode of
    its base parameters, the input `relative_input` follows (main_input, else the first image input; nothing
    connected there = the parent), and the defaults of its own (an absolute format, a Bitmap's image size)."""
    from trafilatura3d.compiler import RELATIVE_INPUT, RELATIVE_PARENT
    from trafilatura3d.package import INPUT_TYPES
    port = d.primary_input() if d.inputs else None
    out: dict[str, Any] = {"mode": RELATIVE_PARENT if d.type in INPUT_TYPES else RELATIVE_INPUT,
                           "from": port or "parent"}
    if port:
        out["unconnected"] = "parent"
    own = getattr(d, "base_defaults", {})
    out.update({k: v for k, v in own.items()})
    if type(d).base_defaults_for is not Node.base_defaults_for:
        out["output_size"] = {"mode": "absolute", "value": "the size of the node's data (its image)"}
    return out


def base_info() -> dict:
    """The base parameters every node and graph has (set_base): label, modes, what the value is."""
    from trafilatura3d.compiler import BASE_PARAMS, MODES
    out = {}
    for name, b in BASE_PARAMS.items():
        out[name] = {"label": b.label, "modes": list(MODES),
                     **({"choices": list(b.choices)} if b.choices else {"value": b.value})}
    return out


# --- edits ------------------------------------------------------------------------------------------

def add_node(graph: Graph, type: str, position=(0.0, 0.0), ref: str = "", params: dict | None = None) -> str:
    """Add a node as the Graph view does: its parameters set, then the project's defaults (normal format)."""
    from trafilatura3d.project import configuration
    new = graph.add_node(type, position, ref=ref)
    for name, value in (params or {}).items():
        graph.set_param(new, name, value)
    d = graph.definition(new)
    fmt = configuration().setting("general", "normal_format")
    if "normal_format" in d.params and "normal_format" not in (params or {}) \
            and fmt in d.params["normal_format"].choices:
        graph.set_param(new, "normal_format", fmt)
    return new


def _endpoint(graph: Graph, ref: str, inputs: bool) -> tuple[str, str]:
    """"node.port" or "node" (its first output, or its primary input) -> (node, port)."""
    node, _, port = ref.rpartition(".") if "." in ref else (ref, "", "")
    if node not in graph.nodes:
        raise DocumentError(f"unknown node {node!r}")
    if not port:
        if inputs:
            port = graph.definition(node).primary_input()
            if port is None:
                raise DocumentError(f"{node}: has no input")
        else:
            port = graph.output_names(node)[0]
    return node, port


def _node(graph: Graph, op: dict) -> str:
    node = op.get("node")
    if node not in graph.nodes:
        raise DocumentError(f"unknown node {node!r}")
    return node


def _apply(graph: Graph, op: dict) -> Any:
    kind = op.get("op")
    if kind == "add_node":
        pos = op.get("position")
        if pos is None:  # right of everything, the layout tidies up
            pos = [max((n.position[0] for n in graph.nodes.values()), default=-COLUMN) + COLUMN, 0.0]
        new = add_node(graph, op["type"], pos, op.get("ref", ""), op.get("params"))
        want = op.get("id")
        if want and want != new:
            if not str(want).replace("_", "").isalnum() or want in graph.nodes:
                raise DocumentError(f"bad or taken node id {want!r}")
            node = graph.nodes.pop(new)
            node.id = want
            graph.nodes[want] = node
            new = want
        return new
    if kind == "remove_node":
        graph.remove_node(_node(graph, op))
    elif kind == "connect":
        src, out = _endpoint(graph, op["from"], inputs=False)
        dst, port = _endpoint(graph, op["to"], inputs=True)
        graph.connect(src, out, dst, port)
    elif kind == "disconnect":
        graph.disconnect(*_endpoint(graph, op["to"], inputs=True))
    elif kind == "set_param":
        node = _node(graph, op)
        for name, value in (op["params"] if "params" in op else {op["param"]: op["value"]}).items():
            linked = graph.nodes[node].links.get(name)
            if linked:  # its value is the graph input's (cook, instances): the node's own is hidden
                raise DocumentError(f"{node}.{name} is driven by graph input {linked!r}: change its default with "
                                    f"set_input(id={linked!r}, default=...), or unlink it with link(node, param)")
            graph.set_param(node, name, value)
    elif kind == "set_base":
        if not op.get("node") and op.get("name") in ("id", "identifier", "label", "description"):
            raise DocumentError(f"{op['name']!r} is not a base parameter: set_graph(id?, label?, description?)")
        node = _node(graph, op) if op.get("node") else None
        graph.set_base(node, op["name"], op["value"])
        if op["name"] == "output_size":  # log2 values: say what they mean
            b = (graph.base if node is None else graph.nodes[node].base)["output_size"]
            x, y = b["value"]
            return {"output_size": f"{1 << x}x{1 << y} px" if b["mode"] == "absolute" else
                    f"{b['mode']} x{2.0 ** x:g} x{2.0 ** y:g} (log2 offsets {x}, {y})"}
    elif kind == "set_graph":
        for key in ("label", "description"):
            if key in op:
                setattr(graph, key, str(op[key]))
        if op.get("id"):
            graph.package.rename_graph(graph.id, op["id"])
    elif kind == "set_input":
        graph.edit_parameter(op["id"], **_input_fields(op, ("id",)))
    elif kind == "insert_dot":
        return insert_dot(graph, *_endpoint(graph, op["to"], inputs=True), op.get("position"))
    elif kind == "move":
        graph.nodes[_node(graph, op)].position = [float(x) for x in op["position"]]
    elif kind == "expose":
        node, pid = _node(graph, op), op.get("id") or op["param"]
        p = graph.expose_node_param(node, op["param"], pid, op.get("label", ""), op.get("group", ""), op.get("kind"),
                                    **_input_fields(op, ("node", "param", "id", "label", "group", "kind")))
        info = {"id": pid, "kind": p.kind.value, "components": p.components(), "editor": p.editor_name()}
        target = graph.definition(node).params[op["param"]].kind
        if target is not p.kind:  # a float driving every component of a vector, through a function
            info["drives"] = f"{node}.{op['param']} ({target.value}) by a function"
        return info
    elif kind == "unexpose":
        graph.unexpose(op["id"])
    elif kind == "link":
        node = _node(graph, op)
        graph.link(node, op["param"], op.get("input"), op.get("component"))
        if op.get("input") and op["param"] not in graph.nodes[node].links:  # a float driving a vector
            return {"drives": f"{node}.{op['param']} by a function"}
    elif kind == "set_output":
        node = _node(graph, op)
        if graph.nodes[node].type != "output":
            raise DocumentError(f"{node}: not an Output node")
        for key in ("identifier", "usage", "label"):
            if key in op:
                graph.set_param(node, key, op[key])
    elif kind == "set_disabled":
        graph.set_disabled([_node(graph, op)], bool(op.get("disabled", True)))
    elif kind == "frame":
        fid = graph.add_item("frame", [0.0, 0.0], **{k: op[k] for k in ("title", "color", "text") if k in op})
        _set_frame_contents(graph, graph.item(fid), op.get("nodes", []), op.get("frames", []))
        return fid
    elif kind == "set_item":
        item = _item(graph, op["id"])
        for key in ("title", "color", "text", "position", "size"):
            if key in op:
                item[key] = list(map(float, op[key])) if key in ("color", "position", "size") else str(op[key])
        if item["kind"] == "frame" and ("nodes" in op or "frames" in op):  # its contents again, fitted around them
            _set_frame_contents(graph, item, op.get("nodes", item.get("nodes", [])),
                                op.get("frames", item.get("frames", [])))
    elif kind == "remove_item":
        item = _item(graph, op["id"])
        for f in _frames(graph):  # the frames in a removed frame go to its parent
            if op["id"] in f.get("frames", []):
                f["frames"] = [c for x in f["frames"] for c in ([x] if x != op["id"] else item.get("frames", []))]
        graph.remove_item(op["id"])
    else:
        raise DocumentError(f"unknown op {kind!r}")
    return None


def _set_frame_contents(graph: Graph, frame: dict, nodes: list, frames: list) -> None:
    """A frame's own nodes and the frames in it, each in one frame only; the frames around it fitted again."""
    nodes = [_node(graph, {"node": k}) for k in nodes]
    if not nodes and not frames:
        raise DocumentError("a frame needs the nodes or frames it groups")
    others = [f for f in _frames(graph) if f is not frame]
    taken = {k for f in others for k in f.get("nodes", [])} & set(nodes)
    if taken:
        raise DocumentError(f"already in a frame: {', '.join(sorted(taken))}")
    parent = frame_parents(graph)
    for c in frames:
        _item(graph, c)
        if c == frame["id"] or graph.item(c)["kind"] != "frame":
            raise DocumentError(f"{c!r}: not a frame that can go in {frame['id']!r}")
        if any(c in f.get("frames", []) for f in others):
            raise DocumentError(f"frame {c!r} is already in a frame")
        p = parent.get(frame["id"])
        while p is not None:  # no frame inside one of its own frames
            if p == c:
                raise DocumentError(f"frame {c!r} holds {frame['id']!r}")
            p = parent.get(p)
    frame["nodes"], frame["frames"] = nodes, list(frames)
    parent, members = frame_parents(graph), frame_members(graph)
    around, p = set(), frame["id"]  # this frame and those around it
    while p is not None and p not in around:
        around.add(p)
        p = parent[p]
    fit_frames(graph, parent, members, around)


def _item(graph: Graph, item_id: str) -> dict:
    try:
        return graph.item(item_id)
    except StopIteration:
        raise DocumentError(f"no item {item_id!r}") from None


def _input_fields(op: dict, taken: tuple) -> dict:
    """The graph input properties of an op (`range` = [min, max])."""
    fields = {k: v for k, v in op.items() if k not in ("op", "range", *taken)}
    if "range" in op:
        fields["min"], fields["max"] = op["range"]
    return fields


OPS = ("add_node", "remove_node", "connect", "disconnect", "set_param", "set_base", "move", "expose", "unexpose",
       "link", "set_output", "set_disabled", "set_graph", "set_input", "insert_dot", "frame", "set_item", "remove_item")


def apply_ops(graph: Graph, ops: list[dict]) -> list[dict]:
    """Apply edits in order, each whole or not at all: a failing one is reported and leaves the graph as it was,
    the next ones go on. Results per op: {"ok": True[, "id": new node]} (expose: the input's id, kind, components,
    editor) or {"error": message}. Raises
    DocumentError when the graph ends up invalid (the caller undoes the batch)."""
    results = []
    for i, op in enumerate(ops):
        before = json.loads(json.dumps(graph.to_dict()))
        try:
            if not isinstance(op, dict):
                raise DocumentError("an op is an object with an 'op' key")
            new = _apply(graph, op)
            results.append({"ok": True, **(new if isinstance(new, dict) else {"id": new} if new else {})})
        except (DocumentError, KeyError, ValueError, TypeError, IndexError) as e:
            graph.load_dict(before)
            msg = f"missing {e}" if isinstance(e, KeyError) else str(e)
            results.append({"error": f"op {i} ({op.get('op') if isinstance(op, dict) else op!r}): {msg}"})
    problems = graph.problems()
    if problems:
        raise DocumentError("; ".join(problems))
    return results


# --- layout -----------------------------------------------------------------------------------------
#
# Layered ("Sugiyama") layout, where the links come first: a graph reads well when its links do.
# 1. Columns: an Output node in column 0 (on the right), any other node one column left of the leftmost node it
#    feeds (a node feeding nothing: column 1, or 0 in a graph without Output nodes).
# 2. A link across several columns gets a waypoint in each column it crosses (a lane of its own: it never runs
#    over a node); with `dots` the waypoints of the links crossing two columns or more become Dot nodes.
# 3. Order in each column: barycentre of the connectors each node is linked to, sweeping back and forth while the
#    crossings go down, then swapping neighbours that cross less; the best order seen is kept.
# 4. Heights: every link as level as possible (least squares on the connector heights, a link between waypoints
#    weighing most, as long links bend least), under the constraint that nothing in a column overlaps: the nodes'
#    real sizes, a constant gap. Nodes snap to the grid, Dots stay on their link.
# 5. Frames: a frame's nodes stay together in every column, the frames in the same order in all columns, a wider gap
#    between two frames' nodes; each frame is fitted around its nodes again.

LANE = 24  # layout: between a waypoint of a long link and what is above or below it
_SWEEPS = 24
_ROUNDS = 12


class _Way:
    """A long link's waypoint in a column it crosses."""
    __slots__ = ("link", "column")

    def __init__(self, link: int, column: int):
        self.link, self.column = link, column


def _columns(graph: Graph, consumers: dict) -> dict[str, int]:
    outputs = any(n.type == "output" for n in graph.nodes.values())
    column: dict[str, int] = {}
    for k in reversed(graph.upstream(list(graph.nodes))):  # consumers first
        if consumers[k]:
            column[k] = 1 + max(column[c] for c in consumers[k])
        else:
            column[k] = 0 if graph.nodes[k].type == "output" or not outputs else 1
    return column


def _isotonic(targets: list[float], weights: list[float], gaps: list[float]) -> list[float]:
    """The heights closest to `targets` (weighted least squares) with y[i + 1] >= y[i] + gaps[i]: pool adjacent
    violators on the heights less the gaps above them."""
    offset, acc = [], 0.0
    for i in range(len(targets)):
        offset.append(acc)
        if i < len(gaps):
            acc += gaps[i]
    blocks: list[list[float]] = []  # [value, weight, count]
    for t, w, o in zip(targets, weights, offset):
        blocks.append([t - o, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    out = []
    for v, _, n in blocks:
        out.extend([v] * n)
    return [v + o for v, o in zip(out, offset)]


def layout(graph: Graph, dots: bool = False) -> dict[str, list[float]]:
    """Arrange the nodes so that the links read well (see above); `dots`: the long links' waypoints become Dot
    nodes. Sets the positions and returns them (the Dots added too)."""
    if not graph.nodes:
        return {}
    parent, members, path = _frame_paths(graph)  # taken before the nodes move
    shapes = {k: _shape(graph, k) for k in graph.nodes}
    consumers: dict[str, list[str]] = {k: [] for k in graph.nodes}
    links = []  # (src, out, dst, port)
    for n in graph.nodes.values():
        for port, (src, out) in n.inputs.items():
            if src in graph.nodes:
                consumers[src].append(n.id)
                links.append((src, out, n.id, port))
    column = _columns(graph, consumers)
    last = max(column.values())
    layers: list[list] = [[] for _ in range(last + 1)]
    for k in graph.nodes:  # column 0: the Output nodes first, in the graph's order
        if graph.nodes[k].type == "output":
            layers[column[k]].append(k)
    for k in graph.nodes:
        if graph.nodes[k].type != "output":
            layers[column[k]].append(k)

    # Segments between adjacent columns: (left item, left connector, right item, right connector, weight).
    def connector(item, name, inputs) -> tuple[int, int]:
        """(index, count) of a connector on its side."""
        if isinstance(item, _Way):
            return 0, 1
        side = shapes[item][2] if inputs else shapes[item][3]
        return (side.index(name) if name in side else 0), max(len(side), 1)

    chains = []
    segments: list[list[tuple]] = [[] for _ in range(last)]  # segments[c]: between column c + 1 and column c
    for i, (src, out, dst, port) in enumerate(links):
        chain = [src] + [_Way(i, c) for c in range(column[src] - 1, column[dst], -1)] + [dst]
        chains.append(chain)
        for a, b in zip(chain, chain[1:]):
            left = connector(a, out, False) if a is src else (0, 1)
            right = connector(b, port, True) if b is dst else (0, 1)
            ways = isinstance(a, _Way) + isinstance(b, _Way)
            c = column[dst] if b is dst else b.column
            segments[c].append((a, left, b, right, (1.0, 2.0, 8.0)[ways]))
        for w in chain[1:-1]:
            layers[w.column].append(w)

    # --- order ---
    def keys(layer) -> dict:
        return {x: i for i, x in enumerate(layer)}

    def at(pos, item, conn) -> float:
        return pos[item] + (conn[0] + 1) / (conn[1] + 1)

    def crossings_between(c, order) -> int:
        left, right = keys(order[c + 1]), keys(order[c])
        pairs = sorted((at(left, a, la), at(right, b, rb)) for a, la, b, rb, _ in segments[c])
        count = 0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                if pairs[i][0] < pairs[j][0] and pairs[i][1] > pairs[j][1]:
                    count += 1
        return count

    def crossings(order) -> int:
        return sum(crossings_between(c, order) for c in range(last))

    def reorder(order, c, toward_right: bool) -> None:
        """Sort column c by the barycentre of its links to column c - 1 (toward_right) or c + 1."""
        if toward_right and c == 0 or not toward_right and c == last:
            return
        other = keys(order[c - 1] if toward_right else order[c + 1])
        sums: dict = {}
        for a, la, b, rb, _ in segments[c - 1 if toward_right else c]:
            mine, theirs = (a, b) if toward_right else (b, a)
            if (column[mine] if not isinstance(mine, _Way) else mine.column) != c:
                continue
            sums.setdefault(mine, []).append(at(other, theirs, rb if toward_right else la))
        current = keys(order[c])
        order[c] = sorted(order[c], key=lambda x: (sum(sums[x]) / len(sums[x]) if x in sums
                                                    else current[x] + 0.5, current[x]))

    order = [list(layer) for layer in layers]
    for c in range(1, last + 1):
        reorder(order, c, True)
    best, best_count = [list(x) for x in order], crossings(order)
    for sweep in range(_SWEEPS):
        if best_count == 0:
            break
        right = sweep % 2 == 0
        for c in (range(1, last + 1) if right else range(last - 1, -1, -1)):
            reorder(order, c, right)
        count = crossings(order)
        if count < best_count:
            best, best_count = [list(x) for x in order], count
    order = best
    improved = best_count > 0
    while improved:  # neighbours swapped where they cross less
        improved = False
        for c in range(last + 1):
            for i in range(len(order[c]) - 1):
                around = [k for k in (c - 1, c) if 0 <= k < last]
                before = sum(crossings_between(k, order) for k in around)
                order[c][i], order[c][i + 1] = order[c][i + 1], order[c][i]
                if sum(crossings_between(k, order) for k in around) < before:
                    improved = True
                else:
                    order[c][i], order[c][i + 1] = order[c][i + 1], order[c][i]

    def frames_of(item) -> list:
        """The frames an item is in, from the outermost."""
        return [] if isinstance(item, _Way) else path.get(item, [])

    if path:  # each frame's nodes (its frames' too) together in each column, the same order in every column
        rank = {x: (i + 0.5) / len(layer) for layer in order for i, x in enumerate(layer)}
        inside: dict[str, list[float]] = {}
        for k, fids in path.items():
            for fid in fids:
                inside.setdefault(fid, []).append(rank[k])
        mean = {fid: sum(v) / len(v) for fid, v in inside.items()}

        def unit_key(x, level: int) -> tuple:
            """Where the unit holding x at this depth goes: its frame (two frames level: by id, never mixed) or x."""
            fs = frames_of(x)
            return (mean[fs[level]], 1, fs[level]) if len(fs) > level else (rank[x], 0, "")

        for layer in order:
            layer.sort(key=lambda x: tuple(unit_key(x, i) for i in range(len(frames_of(x)) + 1)))

    # --- heights ---
    def height(item) -> float:
        return 0.0 if isinstance(item, _Way) else shapes[item][1]

    def gap(item, below=None) -> float:
        """Space under an item: the borders of the frames that end under it and the borders and title bars of those
        that begin above the next one fit in between."""
        base = LANE if isinstance(item, _Way) else GAP
        if below is None:
            return base
        a, b = frames_of(item), frames_of(below)
        same = 0
        while same < min(len(a), len(b)) and a[same] == b[same]:
            same += 1
        ends, begins = len(a) - same, len(b) - same
        if not ends and not begins:
            return base
        return base + ends * FRAME_PAD + begins * (FRAME_PAD + FRAME_BAR) + FRAME_SPACE

    def offset(item, conn) -> float:
        """A connector's height from the item's top."""
        return 0.0 if isinstance(item, _Way) else port_y(conn[0], shapes[item][4])

    y: dict = {}
    for layer in order:
        top = 0.0
        for item, below in zip(layer, layer[1:] + [None]):
            y[item] = top
            top += height(item) + gap(item, below)
    neighbours: dict = {}  # item -> (other item, my connector offset, its offset, weight)
    for c in range(last):
        for a, la, b, rb, w in segments[c]:
            neighbours.setdefault(a, []).append((b, offset(a, la), offset(b, rb), w))
            neighbours.setdefault(b, []).append((a, offset(b, rb), offset(a, la), w))
    for r in range(_ROUNDS):
        for c in (range(last + 1) if r % 2 == 0 else range(last, -1, -1)):
            layer = order[c]
            targets, weights = [], []
            for item in layer:
                nb = neighbours.get(item, [])
                total = sum(w for *_, w in nb)
                if total:
                    targets.append(sum(w * (y[o] + theirs - mine) for o, mine, theirs, w in nb) / total)
                    weights.append(total)
                else:
                    targets.append(y[item])
                    weights.append(1e-3)
            gaps = [height(x) + gap(x, b) for x, b in zip(layer, layer[1:])]
            for item, v in zip(layer, _isotonic(targets, weights, gaps)):
                y[item] = v
    if path:  # frames as bands across the columns they span, so a frame covers nothing but what it holds
        def col(x) -> int:
            return x.column if isinstance(x, _Way) else column[x]

        pref = dict(y)  # the heights found above, pushed down only as much as the bands need

        def place(items, level, floor, ends) -> None:
            """Items in the global order (every column's order is part of it): an item goes under what precedes it
            in its column; a frame (the items in it) starts under what precedes it in all the columns it spans."""
            units: dict = {}
            for x in items:
                fs = frames_of(x)
                units.setdefault(fs[level] if len(fs) > level else x, []).append(x)
            for unit in sorted(units, key=lambda u: unit_key(units[u][0], level)):
                inside = units[unit]
                if len(frames_of(inside[0])) <= level:  # an item
                    c = col(unit)
                    v = max(floor[c], pref[unit])
                    if not isinstance(unit, _Way) and not shapes[unit][4]:  # on the grid here: no pass after
                        v = math.ceil(v / GRID) * GRID
                    y[unit], ends[c] = v, v + height(unit)
                    floor[c] = ends[c] + (LANE if isinstance(unit, _Way) else GAP)
                    continue
                span = range(min(map(col, inside)), max(map(col, inside)) + 1)
                top = max(floor[c] for c in span) + FRAME_PAD + FRAME_BAR
                inner_floor, inner_ends = {c: top for c in span}, {c: -1e9 for c in span}
                place(inside, level + 1, inner_floor, inner_ends)
                bottom = max(inner_ends.values()) + FRAME_PAD
                for c in span:
                    ends[c], floor[c] = bottom, bottom + FRAME_SPACE

        place([x for layer in order for x in layer], 0, {c: -1e9 for c in range(last + 1)},
              {c: -1e9 for c in range(last + 1)})
        lowest = math.floor(min(y.values()) / GRID) * GRID  # a whole number of grid steps: still on it
        for k in y:
            y[k] -= lowest
    lowest = min(y.values())
    for layer in ([] if path else order):  # on the grid, still apart (with frames: done with the bands)
        floor = -1e9
        for item, below in zip(layer, layer[1:] + [None]):
            v = y[item] - lowest
            if not isinstance(item, _Way) and not shapes[item][4]:
                v = max(round(v / GRID) * GRID, math.ceil(floor / GRID) * GRID)
            y[item] = max(v, floor)
            floor = y[item] + height(item) + gap(item, below)

    pos: dict[str, list[float]] = {}
    for c, layer in enumerate(order):
        for item in layer:
            if not isinstance(item, _Way):
                x = -c * COLUMN + ((THUMB + 2 * MARGIN - DOT) / 2 if shapes[item][4] else 0.0)
                pos[item] = [float(x), float(y[item])]
    for k, p in pos.items():
        graph.nodes[k].position = p
    if dots:
        for (src, out, dst, port), chain in zip(links, chains):
            ways = chain[1:-1]
            if len(ways) < 2:
                continue
            keep = [w for i, w in enumerate(ways) if i in (0, len(ways) - 1)
                    or abs(y[w] - y[ways[i - 1]]) > 0.5 or abs(y[w] - y[ways[i + 1]]) > 0.5]
            to, to_port = dst, port
            for w in reversed(keep):
                xy = (-w.column * COLUMN + (THUMB + 2 * MARGIN - DOT) / 2, y[w] - DOT / 2)
                to, to_port = insert_dot(graph, to, to_port, xy), "input"
                pos[to] = list(graph.nodes[to].position)
    fit_frames(graph, parent, members)  # each frame around its nodes and frames again
    return pos


def insert_dot(graph: Graph, dst: str, port: str, position=None) -> str:
    """Put a Dot node on the link into dst.port (the link goes through it); `position`: its top left, default
    halfway along the link. Returns the Dot's id."""
    if port not in graph.nodes[dst].inputs:
        raise DocumentError(f"{dst}.{port}: no link to put a Dot on")
    src, out = graph.nodes[dst].inputs[port]
    if position is None:
        (sw, _, _, souts, sdot), (_, _, dins, _, ddot) = _shape(graph, src), _shape(graph, dst)
        a = graph.nodes[src].position
        b = graph.nodes[dst].position
        ay = a[1] + port_y(souts.index(out) if out in souts else 0, sdot)
        by = b[1] + port_y(dins.index(port) if port in dins else 0, ddot)
        position = ((a[0] + sw + b[0]) / 2 - DOT / 2, (ay + by) / 2 - DOT / 2)
    dot = graph.add_node(DOT_TYPE, [float(position[0]), float(position[1])])
    graph.connect(src, out, dot, "input")
    graph.connect(dot, "output", dst, port)
    return dot


# --- layout quality -------------------------------------------------------------------------------

def _shape(graph: Graph, node_id: str) -> tuple[float, float, list[str], list[str], bool]:
    """(width, height, inputs, outputs, is a Dot) of a node as the Graph view draws it."""
    ins, outs = _ports(graph, node_id)
    dot = graph.nodes[node_id].type == DOT_TYPE
    return (*node_size(len(ins), len(outs), dot), ins, outs, dot)


def _curve(a, b, steps: int = 16) -> list[tuple[float, float]]:
    """A link's curve as a polyline."""
    c1, c2 = link_controls(a, b)
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        pts.append((u ** 3 * a[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * b[0],
                    u ** 3 * a[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * b[1]))
    return pts


def _segments_cross(p, q, r, s) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = orient(r, s, p), orient(r, s, q), orient(p, q, r), orient(p, q, s)
    return d1 * d2 < 0 and d3 * d4 < 0


def _hits_rect(p, q, rect) -> bool:
    """Does segment pq pass through the rectangle (x0, y0, x1, y1)?"""
    x0, y0, x1, y1 = rect
    t0, t1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for d, lo, hi, o in ((dx, x0, x1, p[0]), (dy, y0, y1, p[1])):
        if abs(d) < 1e-9:
            if not lo < o < hi:
                return False
            continue
        a, b = (lo - o) / d, (hi - o) / d
        a, b = min(a, b), max(a, b)
        t0, t1 = max(t0, a), min(t1, b)
        if t0 >= t1:
            return False
    return True


def layout_stats(graph: Graph) -> dict:
    """How readable the graph's placement is, as the Graph view draws it: `overlaps` (pairs of nodes on top of each
    other), `crossings` (pairs of links that cross, links from one connector or into one node not counted) and
    `links_over_nodes` (links passing over a node they don't connect)."""
    shapes = {k: _shape(graph, k) for k in graph.nodes}
    rects = {k: (n.position[0], n.position[1], n.position[0] + shapes[k][0], n.position[1] + shapes[k][1])
             for k, n in graph.nodes.items()}
    keys = list(rects)
    overlaps = sum(1 for i, a in enumerate(keys) for b in keys[i + 1:]
                   if rects[a][0] < rects[b][2] and rects[b][0] < rects[a][2]
                   and rects[a][1] < rects[b][3] and rects[b][1] < rects[a][3])
    links = []
    for n in graph.nodes.values():
        ins = shapes[n.id][2]
        for port, (src, out) in n.inputs.items():
            outs = shapes[src][3] if src in shapes else []
            if port not in ins or out not in outs:
                continue  # a hidden connector
            sp, dp = graph.nodes[src].position, n.position
            a = (sp[0] + shapes[src][0], sp[1] + port_y(outs.index(out), shapes[src][4]))
            b = (dp[0], dp[1] + port_y(ins.index(port), shapes[n.id][4]))
            pts = _curve(a, b)
            box = (min(x for x, _ in pts), min(y for _, y in pts), max(x for x, _ in pts), max(y for _, y in pts))
            links.append(((src, out), n.id, pts, box))
    crossings = 0
    for i, (sa, da, pa, ba) in enumerate(links):
        for sb, db, pb, bb in links[i + 1:]:
            if sa == sb or da == db or ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
                continue
            if any(_segments_cross(pa[k], pa[k + 1], pb[m], pb[m + 1])
                   for k in range(len(pa) - 1) for m in range(len(pb) - 1)):
                crossings += 1
    over = 0
    for (src, _), dst, pts, box in links:
        for k, r in rects.items():
            if k in (src, dst) or r[2] < box[0] or box[2] < r[0] \
                    or r[3] < box[1] or box[3] < r[1]:
                continue
            if any(_hits_rect(pts[m], pts[m + 1], r) for m in range(len(pts) - 1)):
                over += 1
    out = {"overlaps": overlaps, "crossings": crossings, "links_over_nodes": over}
    frames = _frames(graph)
    if frames:  # frames overlapping (not one in the other), nodes over a frame they are not in
        boxes = {f["id"]: _frame_box(f) for f in frames}
        hit = lambda a, b: a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]  # noqa: E731
        _, _, path = _frame_paths(graph)
        holds: dict[str, set] = {fid: set() for fid in boxes}  # a frame's frames, at every depth
        for fids in path.values():
            for i, fid in enumerate(fids):
                holds[fid].update(fids[i + 1:])
        ids = list(boxes)
        out["frame_overlaps"] = sum(1 for i, a in enumerate(ids) for b in ids[i + 1:]
                                    if b not in holds[a] and a not in holds[b] and hit(boxes[a], boxes[b]))
        out["nodes_in_other_frames"] = sum(1 for fid, box in boxes.items() for k, r in rects.items()
                                           if fid not in path.get(k, []) and hit(r, box))
    return out


# --- timings ----------------------------------------------------------------------------------------

def timings(device, compiled, budget_bytes: int = 2 << 30) -> dict:
    """Cook time of each node of a compiled graph (milliseconds, wall clock waiting for the GPU, less the wait for
    an idle GPU), on an evaluator
    of its own (the caller's cache is left alone): one cook of everything compiles the shaders, then each job is
    cooked alone with its inputs ready. Inlined graph instances count under their instance node."""
    from trafilatura3d.evaluator import Evaluator
    ev = Evaluator(device, budget_bytes)
    order, seen = [], set()

    def visit(job):
        if job.key in seen:
            return
        seen.add(job.key)
        for src in job.inputs.values():
            if src:
                visit(src[0])
        order.append(job)

    for job in compiled.jobs.values():
        visit(job)
    def waited(fn) -> float:
        t0 = time.perf_counter()
        fn()
        ev.wait()
        return (time.perf_counter() - t0) * 1000

    try:
        ev.cook(order)
        ev.wait()
        ev.clear()
        idle = min(waited(lambda: None) for _ in range(5))  # waiting for an idle GPU: taken off every node
        per_node: dict[str, float] = {}
        total = 0.0
        for job in order:
            ms = max(0.0, waited(lambda: ev.cook([job])) - idle)
            total += ms
            top = job.path.split("/", 1)[0]
            per_node[top] = per_node.get(top, 0.0) + ms
    finally:
        ev.clear()
    size = [1 << compiled.graph.size[0], 1 << compiled.graph.size[1]]
    return {"size": size, "total_ms": round(total, 2),
            "nodes": {k: round(v, 2) for k, v in sorted(per_node.items(), key=lambda kv: -kv[1])}}
