"""The document: a package (.trafilatura3d, JSON) holding graphs and resources.

A graph holds node instances, its exposed parameters and its own base parameters. A node is an
atomic/library node (a `Node` definition in the registry) or an instance of another graph — of the
same package (`ref: "graph_id"`) or of another package (`ref: "path/to/other.trafilatura3d#graph_id"`,
relative to this package). Graph inputs are its Input color / Input grayscale nodes, its outputs
its Output nodes, its parameters the ones it exposes.

A document may be incomplete while it is edited: unconnected inputs read their default value.
Everything else is always valid: loading refuses broken documents, every edit checks first.
"""

from __future__ import annotations

import copy
import re
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trafilatura3d import shaders
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.base import EDITORS, Input, Node, Output, Param, ParamKind, PortType

FORMAT_VERSION = 2
SUFFIX = ".trafilatura3d"  # a package file
INSTANCE = "graph_instance"
INPUT_TYPES = {"input_color": PortType.COLOR, "input_grayscale": PortType.GRAY, "input_value": PortType.VALUE}
# The graph attributes besides Identifier, Label and Description.
GRAPH_ATTRIBUTES = {"category": "", "author": "", "author_url": "", "tags": "", "group": "", "user_data": "",
                    "physical_size": [0.0, 0.0, 0.0]}

# Templates of a new graph: Empty, then the shaders' (shaders.TEMPLATES: an Output per channel, the 3D View's shader).
TEMPLATES = ("Empty", *shaders.TEMPLATES)

ITEM_DEFAULTS = {"frame": {"title": "Frame", "color": [0.25, 0.45, 0.25, 0.35], "size": [400.0, 260.0], "text": ""},
                 "comment": {"text": "Comment", "size": [220.0, 80.0]},
                 "pin": {"title": "Pin"}}


class DocumentError(ValueError):
    pass


@dataclass
class NodeInstance:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)  # only values that differ from defaults
    base: dict[str, Any] = field(default_factory=dict)  # base parameter overrides (see compiler)
    inputs: dict[str, list[str]] = field(default_factory=dict)  # port -> [source node, source output]
    links: dict[str, str] = field(default_factory=dict)  # param -> exposed graph parameter driving it
    position: list[float] = field(default_factory=lambda: [0.0, 0.0])
    ref: str = ""  # graph instances only
    functions: dict[str, dict] = field(default_factory=dict)  # param -> function graph computing it
    disabled: bool = False  # passes its main input through (Disable node, Shift+D)
    # Graph instances only: output -> "color"/"grayscale" as the source declared it (an imported .sbs node
    # we lack), the type its output gives downstream while its graph can't be resolved.
    outputs: dict[str, str] = field(default_factory=dict)

    def function_graphs(self) -> list[dict]:
        """Every function graph of the node: its base parameters', its parameters' and its FX-Map's."""
        fx = self.params.get("fxmap") if isinstance(self.params.get("fxmap"), dict) else {}
        return ([b["function"] for b in self.base.values() if isinstance(b, dict) and "function" in b]
                + list(self.functions.values())
                + [f for n in fx.get("nodes", []) for f in n.get("functions", {}).values()])

    def to_dict(self) -> dict:
        d = {"id": self.id, "type": self.type}
        for key in ("ref", "params", "base", "inputs", "links", "functions", "disabled", "outputs"):
            if getattr(self, key):
                d[key] = getattr(self, key)
        d["position"] = [round(x, 1) for x in self.position]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NodeInstance:
        return cls(d["id"], d["type"], dict(d.get("params", {})), dict(d.get("base", {})),
                   {k: list(v) for k, v in d.get("inputs", {}).items()}, dict(d.get("links", {})),
                   list(d.get("position", [0.0, 0.0])), d.get("ref", ""), dict(d.get("functions", {})),
                   bool(d.get("disabled", False)), dict(d.get("outputs", {})))


@dataclass
class ExposedParam:
    """A graph parameter: shown on the graph's instances, drives linked node parameters."""
    id: str
    param: Param

    FIELDS = ("label", "min", "max", "group", "description", "editor", "step", "clamp", "user_data", "visible_if")

    def to_dict(self) -> dict:
        p = self.param
        d = {"id": self.id, "kind": p.kind.value, "default": p.default}
        for key in self.FIELDS:
            if not (getattr(p, key) is None or getattr(p, key) is False or getattr(p, key) == ""):  # a 0 is kept
                d[key] = getattr(p, key)
        if p.choices:
            d["choices"] = list(p.choices)
        if p.items:
            d["items"] = [list(i) for i in p.items]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExposedParam:
        kind = ParamKind(d["kind"])
        fields = {k: d[k] for k in cls.FIELDS if k in d}
        fields.setdefault("label", d["id"])
        param = Param(kind, None, choices=tuple(d.get("choices", ())),
                      items=tuple((int(v), str(lbl)) for v, lbl in d.get("items", ())), **fields)
        return cls(d["id"], dataclasses.replace(param, default=param.validate(d["default"])))


class Graph:
    def __init__(self, id: str, package: Package):
        self.id = id
        self.package = package
        self.label = ""
        self.description = ""
        self.attributes: dict[str, Any] = {}  # the other graph attributes (GRAPH_ATTRIBUTES)
        self.presets: list[dict] = []  # embedded presets: {"name", "values": {input id: value}}
        self.base: dict[str, Any] = {}  # the graph's own base parameters (default: relative to parent)
        self.parent_size = [11, 11]  # log2: the application's Parent Size when this graph is the root
        self.items: list[dict] = []  # frames, comments and navigation pins (don't affect cooking)
        self.parameters: dict[str, ExposedParam] = {}
        self.nodes: dict[str, NodeInstance] = {}

    # --- definitions --------------------------------------------------------------------------

    def definition(self, node_id: str) -> Node:
        return self.package.definition(self.nodes[node_id])

    def input_nodes(self) -> list[NodeInstance]:
        return [n for n in self.nodes.values() if n.type in INPUT_TYPES]

    def output_nodes(self) -> list[NodeInstance]:
        return [n for n in self.nodes.values() if n.type == "output"]

    def identifier(self, node: NodeInstance) -> str:
        return self.package.definition(node).values(node)["identifier"]

    # --- topology -----------------------------------------------------------------------------

    def consumers(self, node_id: str) -> list[str]:
        return [n.id for n in self.nodes.values() if any(src == node_id for src, _ in n.inputs.values())]

    def downstream(self, node_id: str) -> set[str]:
        seen, stack = set(), [node_id]
        while stack:
            k = stack.pop()
            if k not in seen:
                seen.add(k)
                stack.extend(self.consumers(k))
        return seen

    def upstream(self, node_ids) -> list[str]:
        """The nodes and everything they depend on, in topological order."""
        order, state = [], {}

        def visit(k):
            if state.get(k) == "done":
                return
            if state.get(k) == "open":
                raise DocumentError(f"{self.id}: cycle through node {k!r}")
            state[k] = "open"
            for src, _ in self.nodes[k].inputs.values():
                visit(src)
            state[k] = "done"
            order.append(k)

        for k in node_ids:
            visit(k)
        return order

    def input_types(self, node_id: str, memo: dict | None = None) -> dict[str, PortType | None]:
        """What is connected to each input of the node: its resolved type, or None (also from a graph instance
        whose reference is missing and whose output types aren't declared: it gives nothing). `memo`: output
        types already resolved (a node feeding several others is resolved once, else the walk is exponential in
        the graph's depth); share one across the calls about the same graph state (a view of every node)."""
        d = self.definition(node_id)
        node = self.nodes[node_id]
        memo = {} if memo is None else memo

        def given(src, output):
            try:
                return self.output_type(src, output, memo)
            except (DocumentError, OSError, ValueError):
                if self.nodes[src].type == INSTANCE:
                    return None
                raise
        return {port: (given(*node.inputs[port]) if port in node.inputs else None) for port in d.inputs}

    def output_type(self, node_id: str, output: str, memo: dict | None = None) -> PortType:
        """The resolved type of a node's output (`memo` as `input_types`'). A graph instance whose reference is
        missing gives the type declared for it (`NodeInstance.outputs`), and raises when there is none."""
        memo = {} if memo is None else memo
        if node_id not in memo:
            node = self.nodes[node_id]
            try:
                d = self.definition(node_id)
            except (DocumentError, OSError, ValueError):
                if node.type != INSTANCE or output not in node.outputs:
                    raise
                memo[node_id] = {o: PortType(t) for o, t in node.outputs.items()}
            else:
                memo[node_id] = d.output_types(self.input_types(node_id, memo), d.values(node))
        return memo[node_id][output]

    def output_names(self, node_id: str) -> list[str]:
        """A node's outputs; for a graph instance whose reference is missing, the outputs declared for it
        (raises when there are none)."""
        node = self.nodes[node_id]
        try:
            return list(self.definition(node_id).outputs)
        except (DocumentError, OSError, ValueError):
            if node.type != INSTANCE or not node.outputs:
                raise
            return list(node.outputs)

    # --- edits --------------------------------------------------------------------------------

    def add_node(self, type: str, position=(0.0, 0.0), ref: str = "") -> str:
        node = NodeInstance("", type, position=list(position), ref=ref)
        self.package.definition(node)  # raises for an unknown type or reference
        base = ref.rsplit("#", 1)[-1] if type == INSTANCE else type
        n = 1
        while f"{base}{n}" in self.nodes:
            n += 1
        node.id = f"{base}{n}"
        if type == "output" or type in INPUT_TYPES:  # identifiers are unique within a graph
            taken = {self.identifier(o) for o in self.nodes.values() if o.type == type}
            stem = "output" if type == "output" else "input"
            k = 1
            while f"{stem}{k}" in taken:
                k += 1
            node.params["identifier"] = f"{stem}{k}"
        self.nodes[node.id] = node
        return node.id

    def remove_node(self, node_id: str) -> None:
        del self.nodes[node_id]
        for n in self.nodes.values():
            for port in [p for p, (src, _) in n.inputs.items() if src == node_id]:
                del n.inputs[port]

    def connect(self, src: str, output: str, dst: str, port: str) -> None:
        """Connect src's output to dst's input port (replacing what was there)."""
        dd = self.definition(dst)
        if output not in self.output_names(src):
            raise DocumentError(f"{src}: has no output {output!r}")
        if port not in dd.inputs:
            raise DocumentError(f"{dst}: has no input {port!r}")
        if port not in dd.visible_inputs(dd.values(self.nodes[dst])):
            raise DocumentError(f"{dst}.{port}: not available, raise the node's input number first")
        got = self.output_type(src, output)
        expected = dd.inputs[port].type
        if not expected.accepts(got):
            raise DocumentError(f"{dst}.{port}: expects {expected.value}, {src}.{output} gives {got.value}")
        if src in self.downstream(dst):
            raise DocumentError(f"connecting {src!r} to {dst!r} would make a cycle")
        old = self.nodes[dst].inputs.get(port)
        self.nodes[dst].inputs[port] = [src, output]
        try:  # a polymorphic change may break typed inputs further down
            self._check_types(self.downstream(dst))
        except DocumentError:
            if old is None:
                del self.nodes[dst].inputs[port]
            else:
                self.nodes[dst].inputs[port] = old
            raise

    def disconnect(self, dst: str, port: str) -> None:
        self.nodes[dst].inputs.pop(port, None)

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        node = self.nodes[node_id]
        param = self.definition(node_id).params.get(name)
        if param is None:
            raise DocumentError(f"{node_id}: has no parameter {name!r}")
        try:
            value = param.validate(value)
        except ValueError as e:
            raise DocumentError(f"{node_id}.{name}: {e}") from None
        if name == "identifier" and (node.type == "output" or node.type in INPUT_TYPES):
            if not value or not value.replace("_", "").isalnum():
                raise DocumentError(f"identifier {value!r}: letters, digits and _ only")
            if any(self.identifier(o) == value for o in self.nodes.values()
                   if o.type == node.type and o.id != node_id):
                raise DocumentError(f"two {node.type} nodes with identifier {value!r}")
        node.params[name] = value
        ports = self.definition(node_id).inputs  # a parameter may change the ports (Pixel processor inputs)
        for port in [p for p in node.inputs if p not in ports]:
            del node.inputs[port]

    def set_base(self, node_id: str | None, name: str, value: Any) -> None:
        """Set a base parameter on a node, or on the graph itself (node_id None)."""
        from trafilatura3d.compiler import BASE_PARAMS
        if name not in BASE_PARAMS:
            raise DocumentError(f"unknown base parameter {name!r} (base parameters: {', '.join(BASE_PARAMS)})")
        try:
            value = BASE_PARAMS[name].validate(value)
        except ValueError as e:
            raise DocumentError(f"{name}: {e}") from None
        (self.base if node_id is None else self.nodes[node_id].base)[name] = value

    def link(self, node_id: str, name: str, exposed: str | None, component: int | str | None = None) -> None:
        """Drive a node parameter by an exposed graph parameter (None: unlink). A float input drives a vector
        parameter through a parameter function: every component (a four-component vector keeps its fourth, the
        alpha), or only `component` (index or x/y/z/w, r/g/b/a), the others kept at their values."""
        if exposed is None:
            self.nodes[node_id].links.pop(name, None)
            return
        d = self.definition(node_id)
        param = d.params.get(name)
        if param is None or exposed not in self.parameters:
            raise DocumentError(f"cannot link {node_id}.{name} to {exposed!r}")
        kind = self.parameters[exposed].param.kind
        n = param.components()
        if kind is param.kind and component is None:
            self.nodes[node_id].links[name] = exposed
            return
        if kind is not ParamKind.FLOAT or param.kind not in (ParamKind.FLOAT2, ParamKind.FLOAT3, ParamKind.FLOAT4):
            raise DocumentError(f"{node_id}.{name} is {param.kind.value}, {exposed!r} is {kind.value}: a link needs "
                                f"the same kind, or a float input driving a float vector")
        if component is None:
            driven = [True] * n if n < 4 else [True, True, True, False]
        else:
            k = "xyzw".find(component) if component in tuple("xyzw") else \
                "rgba".find(component) if component in tuple("rgba") else component
            if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < n:
                raise DocumentError(f"{node_id}.{name} has {n} components: component is 0..{n - 1} or "
                                    f"{', '.join('xyzw'[:n])}, got {component!r}")
            driven = [i == k for i in range(n)]
        self.nodes[node_id].links.pop(name, None)
        self.set_function(node_id, name, _driven_by(exposed, d.values(self.nodes[node_id])[name], driven))

    def can_disable(self, node_id: str) -> bool:
        """The rule: at least one input, exactly one output, main input type = output type."""
        d = self.definition(node_id)
        port = d.primary_input()
        if port is None or len(d.outputs) != 1:
            return False
        out = next(iter(d.outputs.values())).type
        return d.inputs[port].type is out or PortType.ANY in (d.inputs[port].type, out)

    def set_disabled(self, node_ids, disabled: bool) -> None:
        bad = [k for k in node_ids if not self.can_disable(k)]
        if bad:
            raise DocumentError(f"can't disable {', '.join(bad)}: needs an input and one output of the same type")
        for k in node_ids:
            self.nodes[k].disabled = disabled

    def add_item(self, kind: str, position, **fields) -> str:
        """A graph item: a frame (title, color, size), a comment (text) or a navigation pin (title)."""
        if kind not in ("frame", "comment", "pin"):
            raise DocumentError(f"unknown graph item {kind!r}")
        n, ids = 1, {i["id"] for i in self.items}
        while f"{kind}{n}" in ids:
            n += 1
        item = {"id": f"{kind}{n}", "kind": kind, "position": list(position), **ITEM_DEFAULTS[kind], **fields}
        self.items.append(item)
        return item["id"]

    def item(self, item_id: str) -> dict:
        return next(i for i in self.items if i["id"] == item_id)

    def remove_item(self, item_id: str) -> None:
        self.items = [i for i in self.items if i["id"] != item_id]

    def function_variables(self) -> dict[str, str]:
        """What parameter functions of this graph's nodes can Get: system variables and the
        graph's exposed parameters."""
        from trafilatura3d.functions import BUILTINS, PARAM_TYPES
        values = {self.identifier(n): "float4" for n in self.nodes.values() if n.type == "input_value"}
        return {**BUILTINS, **values, **{pid: PARAM_TYPES[p.param.kind] for pid, p in self.parameters.items()
                                         if p.param.kind in PARAM_TYPES}}

    def set_function(self, node_id: str, name: str, function: dict | None) -> None:
        """Drive a node parameter by a function graph (None removes it)."""
        from trafilatura3d.functions import PARAM_TYPES, FunctionError, FunctionGraph, set_variables
        if function is None:
            self.nodes[node_id].functions.pop(name, None)
            return
        param = self.definition(node_id).params.get(name)
        if param is None or param.kind not in PARAM_TYPES:
            raise DocumentError(f"{node_id}.{name} can't be driven by a function")
        try:
            others = [f for k, f in self.nodes[node_id].functions.items() if k != name]
            fg = FunctionGraph(function, self.function_variables(), set_variables(others))
        except FunctionError as e:
            raise DocumentError(f"{node_id}.{name}: {e}") from None
        if fg.types[fg.output] != PARAM_TYPES[param.kind]:
            raise DocumentError(f"{node_id}.{name} needs a {PARAM_TYPES[param.kind]}, the function gives "
                                f"{fg.types[fg.output]}")
        self.nodes[node_id].functions[name] = function

    def expose(self, id: str, param: Param) -> None:
        """Add or replace a graph input (every property of `param` kept, default validated)."""
        if not id or not id.replace("_", "").isalnum():
            raise DocumentError(f"parameter id {id!r}: letters, digits and _ only")
        try:
            default = param.validate(param.default)
        except ValueError as e:
            raise DocumentError(f"{id}: {e}") from None
        self.parameters[id] = ExposedParam(id, dataclasses.replace(param, default=default, label=param.label or id))

    def expose_node_param(self, node_id: str, name: str, id: str, label: str = "", group: str = "",
                          kind: str | None = None, **fields) -> Param:
        """Expose a node parameter as a new graph input `id` (its current value the default, `fields` other
        properties of the input: min, max, editor, description...) and link it; returns the input's parameter.
        `kind` "float" on a vector parameter makes a one-number input that drives every component through a
        parameter function (a four-component vector keeps its fourth, the alpha, as it is)."""
        d = self.definition(node_id)
        if name not in d.params:
            raise DocumentError(f"{node_id}: has no parameter {name!r}")
        if id in self.parameters:
            raise DocumentError(f"graph input {id!r} exists already: link the parameter to it, or choose another id")
        param, value = d.params[name], d.values(self.nodes[node_id])[name]
        if kind in (None, param.kind.value):
            self.expose(id, self._with_fields(dataclasses.replace(
                param, default=value, label=label or id, group=group), fields))
            self.link(node_id, name, id)
            return self.parameters[id].param
        vectors = (ParamKind.FLOAT2, ParamKind.FLOAT3, ParamKind.FLOAT4)
        if kind != ParamKind.FLOAT.value or param.kind not in vectors:
            also = " or float (one value for every component)" if param.kind in vectors else ""
            raise DocumentError(f"{node_id}.{name} is {param.kind.value}: it is exposed as {param.kind.value}{also}")
        self.expose(id, self._with_fields(Param(ParamKind.FLOAT, float(value[0]), label or id, param.min, param.max,
                                                group=group), fields))
        self.link(node_id, name, id)
        return self.parameters[id].param

    def edit_parameter(self, id: str, **fields) -> None:
        """Change properties of a graph input: default, label, min, max, group, description..."""
        if id not in self.parameters:
            raise DocumentError(f"no graph input {id!r} (inputs: {', '.join(self.parameters) or 'none'})")
        self.expose(id, self._with_fields(self.parameters[id].param, fields))

    @staticmethod
    def _with_fields(param: Param, fields: dict) -> Param:
        allowed = {"default", "choices", "items", *ExposedParam.FIELDS}
        bad = sorted(set(fields) - allowed)
        if bad:
            raise DocumentError(f"unknown graph input properties {bad} (properties: {', '.join(sorted(allowed))})")
        editors = EDITORS.get(param.kind, ())
        if fields.get("editor") and fields["editor"] not in editors:
            raise DocumentError(f"editor {fields['editor']!r} is not one of a {param.kind.value} input's: "
                                f"{', '.join(editors) or 'none'}")
        if "choices" in fields:
            fields = {**fields, "choices": tuple(fields["choices"])}
        if "items" in fields:
            fields = {**fields, "items": tuple((int(v), str(lbl)) for v, lbl in fields["items"])}
        return dataclasses.replace(param, **fields)

    def rename_parameter(self, old: str, new: str) -> None:
        """Change an input's identifier, following its links and Visible if references."""
        if new == old:
            return
        if not new or not new.replace("_", "").isalnum() or new in self.parameters:
            raise DocumentError(f"bad or taken parameter id {new!r}")
        self.parameters = {(new if k == old else k): (ExposedParam(new, v.param) if k == old else v)
                           for k, v in self.parameters.items()}
        for n in self.nodes.values():
            for name, pid in list(n.links.items()):
                if pid == old:
                    n.links[name] = new
        pattern = re.compile(r"(input\.)" + re.escape(old) + r"\b|(input\[[\"'])" + re.escape(old) + r"([\"']\])")
        sub = lambda s: pattern.sub(lambda m: (m.group(1) or m.group(2)) + new + (m.group(3) or ""), s)
        for pid, ep in self.parameters.items():
            if ep.param.visible_if:
                self.parameters[pid] = ExposedParam(pid, dataclasses.replace(ep.param, visible_if=sub(ep.param.visible_if)))
        for n in self.nodes.values():
            if n.params.get("visible_if"):
                n.params["visible_if"] = sub(n.params["visible_if"])

    def move_parameter(self, pid: str, delta: int) -> None:
        """Reorder the graph's inputs (their order on instances)."""
        keys = list(self.parameters)
        i = keys.index(pid)
        j = max(0, min(len(keys) - 1, i + delta))
        keys.insert(j, keys.pop(i))
        self.parameters = {k: self.parameters[k] for k in keys}

    def move_node(self, node_id: str, delta: int, kind_filter) -> None:
        """Reorder Input or Output nodes among themselves (their connector order on instances)."""
        same = [k for k, n in self.nodes.items() if kind_filter(n)]
        i = same.index(node_id)
        j = max(0, min(len(same) - 1, i + delta))
        if i == j:
            return
        other = same[j]
        keys = list(self.nodes)
        a, b = keys.index(node_id), keys.index(other)
        keys[a], keys[b] = keys[b], keys[a]
        self.nodes = {k: self.nodes[k] for k in keys}

    def unused_parameters(self) -> list[str]:
        """Inputs linked to no node parameter and read by no function (Clean Inputs)."""
        used = {pid for n in self.nodes.values() for pid in n.links.values()}
        text = json.dumps([n.functions for n in self.nodes.values()]) + json.dumps(
            [n.params.get("per_pixel_function") for n in self.nodes.values()])
        return [pid for pid in self.parameters if pid not in used and f'"{pid}"' not in text]

    def unexpose(self, id: str) -> None:
        self.parameters.pop(id, None)
        for n in self.nodes.values():
            for name in [k for k, v in n.links.items() if v == id]:
                del n.links[name]

    # --- validation ---------------------------------------------------------------------------

    def problems(self) -> list[str]:
        problems = []
        from trafilatura3d.compiler import BASE_PARAMS
        for n in self.nodes.values():
            if n.outputs and (n.type != INSTANCE or not set(n.outputs.values()) <= {"color", "grayscale"}):
                problems.append(f"{self.id}.{n.id}: bad declared output types {n.outputs!r}")
            try:
                d = self.package.definition(n)
            except (DocumentError, OSError, ValueError) as e:
                # A missing referenced package or graph is a warning (trafilatura3d.diagnostics), not a reason
                # to refuse the package: it opens and shows where to repair.
                if n.type != INSTANCE:
                    problems.append(f"{self.id}.{n.id}: {e}")
                continue
            for name, value in n.params.items():
                if name not in d.params:
                    problems.append(f"{self.id}.{n.id}: has no parameter {name!r}")
                    continue
                try:
                    n.params[name] = d.params[name].validate(value)
                except ValueError as e:
                    problems.append(f"{self.id}.{n.id}.{name}: {e}")
            for name, value in n.base.items():
                try:
                    n.base[name] = BASE_PARAMS[name].validate(value)
                except (KeyError, ValueError) as e:
                    problems.append(f"{self.id}.{n.id}: base parameter {name!r}: {e}")
            for name, exposed in n.links.items():
                if name not in d.params or exposed not in self.parameters:
                    problems.append(f"{self.id}.{n.id}: bad link {name!r} -> {exposed!r}")
            from trafilatura3d.functions import PARAM_TYPES
            for name, function in n.functions.items():
                # Structure only: a function being built may be incomplete; types are checked when compiled.
                if name not in d.params or d.params[name].kind not in PARAM_TYPES \
                        or not isinstance(function, dict) or not isinstance(function.get("nodes"), list):
                    problems.append(f"{self.id}.{n.id}: bad function on {name!r}")
            for port, (src, output) in n.inputs.items():
                if port not in d.inputs:
                    problems.append(f"{self.id}.{n.id}: has no input {port!r}")
                elif src not in self.nodes:
                    problems.append(f"{self.id}.{n.id}.{port}: unknown node {src!r}")
        if problems:
            return problems
        try:
            self.upstream(self.nodes)
            self._check_types(self.nodes)
        except DocumentError as e:
            problems.append(str(e))
        for kind in ["output", *INPUT_TYPES]:
            ids = [self.identifier(n) for n in self.nodes.values() if n.type == kind]
            for dup in sorted({x for x in ids if ids.count(x) > 1}):
                problems.append(f"{self.id}: two {kind} nodes with identifier {dup!r}")
        return problems

    def _check_types(self, node_ids) -> None:
        memo: dict = {}
        for k in node_ids:
            n = self.nodes[k]
            try:
                d = self.definition(k)
            except (DocumentError, OSError, ValueError):
                if n.type == INSTANCE:  # a missing referenced graph: a warning, links not checkable
                    continue
                raise
            for port, (src, output) in n.inputs.items():
                try:
                    outputs = self.output_names(src)
                except (DocumentError, OSError, ValueError):
                    if self.nodes[src].type == INSTANCE:  # a missing referenced graph, nothing declared
                        continue
                    raise
                if output not in outputs:
                    raise DocumentError(f"{self.id}.{k}.{port}: {src} has no output {output!r}")
                expected = d.inputs[port].type
                got = self.output_type(src, output, memo)
                if not expected.accepts(got):
                    raise DocumentError(f"{self.id}.{k}.{port}: expects {expected.value}, "
                                        f"{src}.{output} gives {got.value}")

    # --- file ---------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {}
        if self.label:
            d["label"] = self.label
        if self.description:
            d["description"] = self.description
        d["parent_size"] = list(self.parent_size)
        if self.base:
            d["base"] = self.base
        if self.parameters:
            d["parameters"] = [p.to_dict() for p in self.parameters.values()]
        d["nodes"] = [n.to_dict() for n in self.nodes.values()]
        if self.items:
            d["items"] = self.items
        if self.attributes:
            d["attributes"] = self.attributes
        if self.presets:
            d["presets"] = self.presets
        return d

    def load_dict(self, d: dict) -> None:
        self.label = d.get("label", "")
        self.description = d.get("description", "")
        self.attributes = {k: v for k, v in d.get("attributes", {}).items() if k in GRAPH_ATTRIBUTES}
        self.presets = [{"name": str(p["name"]), "values": dict(p.get("values", {}))} for p in d.get("presets", [])]
        self.base = dict(d.get("base", {}))
        self.parent_size = list(d.get("parent_size", [11, 11]))
        self.items = [dict(i) for i in d.get("items", []) if i.get("kind") in ITEM_DEFAULTS]
        self.parameters = {p["id"]: ExposedParam.from_dict(p) for p in d.get("parameters", [])}
        nodes = [NodeInstance.from_dict(n) for n in d.get("nodes", [])]
        self.nodes = {n.id: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise DocumentError(f"{self.id}: duplicate node ids")


@dataclass(frozen=True)
class Note:
    """A message about the document that is not saved with it (what an import could not convert), shown with
    the graph's warnings."""
    graph: str
    node: str | None
    message: str


@dataclass
class Resource:
    kind: str  # "bitmap" or "scene" (a 3D scene of the 3D View)
    path: str  # relative to the package ("primitive:<name>" for a scene on a primitive)
    state: dict | None = None  # a scene's 3D View state (camera, lights, materials, renderer)
    baking: dict | None = None  # a scene's baking configuration (Bake model information)


class Package:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.graphs: dict[str, Graph] = {}
        self.resources: dict[str, Resource] = {}
        self.notes: list[Note] = []
        self._instance_defs: dict[str, Node] = {}
        self._external: dict[Path, Package] = {}

    # --- graphs -------------------------------------------------------------------------------

    def new_graph(self, id: str, size_log2: int = 11) -> Graph:
        """A new graph, `size_log2` its Parent Size (the New graph dialog's Size)."""
        if not id or not id.replace("_", "").isalnum() or id in self.graphs:
            raise DocumentError(f"bad or taken graph id {id!r}")
        g = Graph(id, self)
        g.parent_size = [size_log2, size_log2]
        self.graphs[id] = g
        return g

    def remove_graph(self, id: str) -> None:
        del self.graphs[id]

    def rename_graph(self, old: str, new: str) -> None:
        """Change a graph's identifier, in place in the package's order, following its instances in the package."""
        if new == old:
            return
        if not new or not new.replace("_", "").isalnum() or new in self.graphs:
            raise DocumentError(f"bad or taken graph id {new!r}")
        g = self.graphs[old]
        self.graphs = {(new if k == old else k): v for k, v in self.graphs.items()}
        g.id = new
        for other in self.graphs.values():
            for n in other.nodes.values():
                if n.type == INSTANCE and n.ref == old:
                    n.ref = new

    # --- node definitions ---------------------------------------------------------------------

    def definition(self, node: NodeInstance) -> Node:
        if node.type != INSTANCE:
            if node.type not in NODES:
                raise DocumentError(f"unknown node type {node.type!r}")
            d = NODES[node.type]
            return d.instantiate(d.values(node))
        return InstanceDefinition(*self.resolve(node.ref))

    def resolve(self, ref: str) -> tuple[Package, Graph]:
        """The package and graph a graph instance refers to."""
        if "#" in ref:
            rel, gid = ref.rsplit("#", 1)
            if rel.startswith("sbs://") or rel.endswith((".sbs", ".sbsar")):  # an imported .sbs node we lack
                raise DocumentError(f"{rel}#{gid}: a .sbs node with no Trafilatura 3D counterpart")
            from trafilatura3d.project import alias_of, expand
            if alias_of(rel):  # alias://path from the project configuration
                try:
                    path = Path(expand(rel)).resolve()
                except KeyError:
                    raise DocumentError(f"Check the alias '{alias_of(rel)}' is defined in your project") from None
            elif self.path is None:
                raise DocumentError(f"save the package before referencing {rel!r}")
            else:
                path = (self.path.parent / rel).resolve()
            if path not in self._external:
                self._external[path] = Package.load(path)
            pkg = self._external[path]
        else:
            pkg, gid = self, ref
        if gid not in pkg.graphs:
            raise DocumentError(f"unknown graph {ref!r}")
        return pkg, pkg.graphs[gid]

    # --- validation ---------------------------------------------------------------------------

    def validate(self) -> None:
        problems = [p for g in self.graphs.values() for p in g.problems()]
        if problems:
            raise DocumentError("invalid package:\n  " + "\n  ".join(problems))

    # --- file ---------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {"trafilatura3d": FORMAT_VERSION, "graphs": {gid: g.to_dict() for gid, g in self.graphs.items()}}
        if self.resources:
            d["resources"] = {rid: {"kind": r.kind, "path": r.path, **({"state": r.state} if r.state else {}),
                                    **({"baking": r.baking} if r.baking else {})}
                              for rid, r in self.resources.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict, path: Path | None = None) -> Package:
        if data.get("trafilatura3d") != FORMAT_VERSION:
            raise DocumentError(f"not a trafilatura3d v{FORMAT_VERSION} package")
        pkg = cls(path)
        for gid, gd in data.get("graphs", {}).items():
            g = Graph(gid, pkg)
            g.load_dict(copy.deepcopy(gd))
            pkg.graphs[gid] = g
        pkg.resources = {rid: Resource(r["kind"], r["path"], r.get("state"), r.get("baking"))
                         for rid, r in data.get("resources", {}).items()}
        pkg.validate()
        return pkg

    def restore(self, data: dict) -> None:
        """Replace the content with a snapshot from `to_dict()` (undo/redo), keeping this object."""
        other = Package.from_dict(data, self.path)
        self.graphs = other.graphs
        for g in self.graphs.values():
            g.package = self
        self.resources = other.resources

    def restore_layout(self, data: dict) -> None:
        """Put back the arrangement of a snapshot from `to_dict()` that differs from this content only there (positions
        of nodes, frames, comments and pins, also inside function and FX-Map graphs), keeping the graph and node
        objects: who holds them stays valid."""
        other = Package.from_dict(data, self.path)
        for gid, g in self.graphs.items():
            o = other.graphs[gid]
            g.items = o.items
            for nid, n in g.nodes.items():
                m = o.nodes[nid]
                n.position, n.params, n.base, n.functions = m.position, m.params, m.base, m.functions

    @classmethod
    def load(cls, path: str | Path) -> Package:
        path = Path(path).resolve()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path)

    def save(self, path: str | Path | None = None) -> None:
        """Write the package, creating the folders missing on the way; plugins hear before_save (and may refuse it
        with Veto) and package_saved. The package takes the new path only once it is written."""
        from trafilatura3d import plugins
        target = Path(path).resolve() if path is not None else self.path
        plugins.emit("before_save", package=self, path=target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.path = target
        plugins.emit("package_saved", package=self, path=self.path)


def _driven_by(input_id: str, value: list, driven: list[bool]) -> dict:
    """A parameter function: a float vector whose `driven` components are the float graph input `input_id` and the
    others their constant in `value`, joined left to right by Vector nodes."""
    nodes = [{"id": "get", "type": "get_float", "params": {"variable": input_id}, "position": [0, 0]}]
    parts = []
    for i, on in enumerate(driven):
        if on:
            parts.append("get")
        else:
            nodes.append({"id": f"c{i}", "type": "const_float", "params": {"value": float(value[i])},
                          "position": [0, 80 * (i + 1)]})
            parts.append(f"c{i}")
    out = parts[0]
    for i, part in enumerate(parts[1:], 2):
        nodes.append({"id": f"v{i}", "type": f"vector_float{i}", "inputs": {"input_1": out, "input_2": part},
                      "position": [160 * (i - 1), 0]})
        out = f"v{i}"
    return {"nodes": nodes, "output": out}


def apply_template(graph: Graph, name: str) -> shaders.Template | None:
    """A template's nodes (TEMPLATES) in a graph: per channel of its shader an Output with the channel's usage, fed
    by a Uniform color at the channel's neutral value (the mesh shows a plain material, not black). Returns the
    shader template (None: Empty), whose shader the 3D View takes."""
    if name not in TEMPLATES:
        raise DocumentError(f"unknown template {name!r} (templates: {', '.join(TEMPLATES)})")
    t = shaders.TEMPLATES.get(name)
    for i, usage in enumerate(t.channels if t else ()):
        gray, value = shaders.channel_gray(t.shader, usage), shaders.neutral(t.shader, usage)
        u = graph.add_node("uniform_color", (160, i * 160))
        graph.set_param(u, "color_mode", "Grayscale" if gray else "Color")
        graph.set_param(u, "output_color", list(value))
        o = graph.add_node("output", (480, i * 160))
        graph.set_param(o, "identifier", usage.lower())
        graph.set_param(o, "usage", usage)
        graph.connect(u, "output", o, "input")
    return t


class InstanceDefinition(Node):
    """A graph used as a node: its Input nodes are the inputs, its Output nodes the outputs, its
    exposed parameters the parameters. Cooking inlines the graph (see compiler)."""

    type = INSTANCE
    category = "Graph instances"

    def __init__(self, package: Package, graph: Graph):
        self.package, self.graph = package, graph
        self.label = graph.label or graph.id
        self.inputs = {graph.identifier(n): Input(INPUT_TYPES[n.type]) for n in graph.input_nodes()}
        self.outputs = {graph.identifier(n): Output(PortType.ANY) for n in graph.output_nodes()}
        self.params = {pid: p.param for pid, p in graph.parameters.items()}

    def _io_visible(self, nodes, values) -> list[str]:
        from trafilatura3d.visibility import visible
        shown = []
        for n in nodes:
            v = self.package.definition(n).values(n)
            if visible(v.get("visible_if", ""), {pid: values.get(pid, p.default) for pid, p in self.params.items()}):
                shown.append(self.graph.identifier(n))
        return shown

    def visible_inputs(self, values):
        return self._io_visible(self.graph.input_nodes(), values)

    def visible_outputs(self, values):
        return self._io_visible(self.graph.output_nodes(), values)

    def output_types(self, input_types, values):
        # An output's type is what reaches its Output node inside the graph.
        return {self.graph.identifier(n): (self.graph.output_type(*n.inputs["input"])
                                           if "input" in n.inputs else PortType.GRAY)
                for n in self.graph.output_nodes()}
