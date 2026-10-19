"""Compiles a graph of a package into jobs for the evaluator.

Graph instances are inlined (their Input nodes take what is connected to the instance, their
Output nodes give the instance's outputs), parameters are resolved (defaults, links to exposed
parameters, the instance's values) and every node gets concrete base parameters.

Base parameters (inheritance):
each is {"mode": absolute | relative_input | relative_parent, "value": ...}. Relative to input
inherits from what is connected to the node's primary input, relative to parent from the graph
(whose own parent is the instance node using it, or for a root graph the application's Parent
Size). Defaults: nodes and graph instances relative to input, Input nodes and graphs relative to
parent, Bitmap's output size absolute (a node may declare its own `base_defaults`).
Output Size is log2 (absolute: the exponent; relative: added to it); Random Seed relative is added;
Pixel Size relative multiplies. Output Size, Pixel Size and Random Seed of a node may be computed by a function
("function"), evaluated before inheriting. A node with nothing on its primary input resolves
"relative to input" from its parent (the documentation leaves this open).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from trafilatura3d.evaluator import PREVIEW, Job
from trafilatura3d.nodes.base import Param, ParamKind, PortType
from trafilatura3d.package import INPUT_TYPES, INSTANCE, DocumentError, Graph, Package

MAX_LOG2 = 13  # 8192 px
SIZE_LIMIT_LOG2 = 13  # the Cooking size limit (8192 px); the editor's Preferences set it

RELATIVE_PARENT, RELATIVE_INPUT, ABSOLUTE = "relative_parent", "relative_input", "absolute"
MODES = (RELATIVE_PARENT, RELATIVE_INPUT, ABSOLUTE)
DEPTHS = {"8": "8 Bits per Channel", "16": "16 Bits per Channel", "16f": "HDR Low Precision (16F)",
          "32f": "HDR High Precision (32F)"}
TILINGS = {"none": 0, "h": 1, "v": 2, "hv": 3}  # the $tiling values
TILING_LABELS = {"none": "No Tiling", "h": "Horizontal Tiling", "v": "Vertical Tiling", "hv": "H and V Tiling"}
PIXEL_RATIOS = ("stretch", "square")


@dataclass(frozen=True)
class BaseParam:
    """A base parameter value is {"mode": ..., "value": ...}, with an optional "function" (a function graph giving
    the value, as functions on base parameters: a library pattern's Output Size from its Tiling) where
    the parameter has a `kind`."""
    label: str
    check: Any  # value -> normalized value, raises ValueError
    kind: ParamKind | None = None  # what a function gives
    choices: tuple = ()  # the values it takes, when it is a choice
    value: str = ""  # what its value is, otherwise (for a description of the parameter)

    def validate(self, v: Any) -> dict:
        if not isinstance(v, dict) or v.get("mode") not in MODES:
            raise ValueError(f"expected {{'mode': one of {list(MODES)}, 'value': ...}}, got {v!r}")
        out = {"mode": v["mode"], "value": self.check(v["value"])}
        if "function" in v:
            f = v["function"]
            if self.kind is None or not isinstance(f, dict) or not isinstance(f.get("nodes"), list):
                raise ValueError(f"bad function {f!r}")
            out["function"] = f
        return out


def _int2(v):
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) and -12 <= x <= MAX_LOG2 for x in v):
        return [int(x) for x in v]
    raise ValueError(f"expected two integers in -12..{MAX_LOG2}, got {v!r}")


def _float2(v):
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) and x > 0 for x in v):
        return [float(x) for x in v]
    raise ValueError(f"expected two positive numbers, got {v!r}")


def _choice(options):
    def check(v):
        if v not in options:
            raise ValueError(f"expected one of {list(options)}, got {v!r}")
        return v
    return check


def _int(v):
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    raise ValueError(f"expected an integer, got {v!r}")


BASE_PARAMS = {
    "output_size": BaseParam("Output Size", _int2, ParamKind.INT2,
                             value=f"[log2 width, log2 height]: relative modes add to the reference's, absolute is "
                                   f"the size itself (11 = 2048), -12..{MAX_LOG2}"),
    "output_format": BaseParam("Output Format", _choice(tuple(DEPTHS)), choices=tuple(DEPTHS)),
    "pixel_size": BaseParam("Pixel Size", _float2, ParamKind.FLOAT2, value="[width, height], positive"),
    "pixel_ratio": BaseParam("Pixel Ratio", _choice(PIXEL_RATIOS), choices=PIXEL_RATIOS),
    "tiling": BaseParam("Tiling Mode", _choice(tuple(TILINGS)), choices=tuple(TILINGS)),
    "seed": BaseParam("Random Seed", _int, ParamKind.INT, value="an integer"),
}
_RELATIVE_ZERO = {"output_size": [0, 0], "output_format": "8", "pixel_size": [1.0, 1.0],
                  "pixel_ratio": "square", "tiling": "hv", "seed": 0}
NODE_DEFAULTS = {k: {"mode": RELATIVE_INPUT, "value": v} for k, v in _RELATIVE_ZERO.items()}
PARENT_DEFAULTS = {k: {"mode": RELATIVE_PARENT, "value": v} for k, v in _RELATIVE_ZERO.items()}


@dataclass(frozen=True)
class Context:
    """Concrete base parameters (what a node resolves to, and what children inherit)."""
    size: tuple[int, int]  # log2
    depth: str
    pixel_size: tuple[float, float]
    pixel_ratio: str
    tiling: str
    seed: int
    # The preview's (a dragged value): the textures are computed this many powers of 2 below `size`, which keeps
    # the size they stand for (what functions read as $size, what children inherit).
    shift: int = 0


def application_context(parent_size_log2=(11, 11), shift: int = 0) -> Context:
    """The parent of a root graph: the Parent Size of the Graph view, 8 bit, tiled, seed 0."""
    return Context(tuple(parent_size_log2), "8", (1.0, 1.0), "square", "hv", 0, shift)


@dataclass
class Compiled:
    jobs: dict[str, Job]  # node path -> job (inlined instance nodes as "instance/node")
    nodes: dict[str, dict[str, tuple[Job, str]]]  # top-level node id -> output -> (job, output)
    outputs: dict[str, tuple[Job, str]]  # Output node identifier -> its job
    graph: Context  # the root graph's resolved base parameters


def compile_graph(package: Package, graph_id: str, parent: Context | None = None, evaluator=None,
                  values: dict | None = None, shift: int = 0) -> Compiled:
    """Compile a root graph; `parent` defaults to the application with the graph's Parent Size.
    With an `evaluator`, values reaching Input value nodes of instanced graphs are read back so their
    functions can Get them (without one, they read the Input value's default). `values` override the
    graph's input parameters (the Properties' Preview mode, a CLI render). `shift`: every texture computed that
    many powers of 2 smaller (a dragged value's preview, `Context.shift`), in the evaluator's preview partition."""
    graph = package.graphs[graph_id]
    parent = replace(parent or application_context(graph.parent_size), shift=shift)
    ctx = _resolve({**PARENT_DEFAULTS, **graph.base}, parent, None)
    jobs: dict[str, Job] = {}
    nodes, outputs = _compile(package, graph, dict(values or {}), {}, ctx, "", jobs, top=True, ev=evaluator)
    return Compiled(jobs, nodes, outputs, ctx)


def _compile(package: Package, graph: Graph, values: dict, inputs: dict, ctx: Context, prefix: str,
             jobs: dict, top: bool, ev=None):
    """Returns (node id -> output -> source, output identifier -> source). `values`: exposed
    parameter values set by the instance; `inputs`: input identifier -> source or None."""
    from trafilatura3d.functions import from_param, set_variables
    exposed = {pid: values.get(pid, p.param.default) for pid, p in graph.parameters.items()}
    # What functions of this graph can Get, besides system variables: name -> (kind, function value).
    fvars = {pid: (graph.parameters[pid].param.kind.value, from_param(v, graph.parameters[pid].param))
             for pid, v in exposed.items()}
    results: dict[str, dict[str, tuple[Job, str]]] = {}
    outputs: dict[str, tuple[Job, str]] = {}
    for node_id in graph.upstream(graph.nodes):
        node = graph.nodes[node_id]
        try:
            d = package.definition(node)
        except (DocumentError, OSError, ValueError):
            if node.type != INSTANCE:
                raise
            results[node_id] = {}  # a missing referenced graph: downstream reads it as unconnected
            continue
        vals = d.values(node)
        for name, pid in node.links.items():
            vals[name] = exposed[pid]
        # A source a pass-through dropped (disabled node with nothing on its input) reads as unconnected.
        srcs = {port: results[src][out] for port, (src, out) in node.inputs.items() if out in results.get(src, {})}
        if getattr(d, "passthrough", False) or (node.disabled and graph.can_disable(node_id)):
            port, out = d.primary_input(), next(iter(d.outputs))
            results[node_id] = {out: srcs[port]} if port in srcs else {}
            continue
        path = prefix + node_id

        if node.type == "input_value":
            source = None if top else inputs.get(vals["identifier"])
            value = list(vals["default_value"])
            if source is not None and ev is not None:
                value = [float(x) for x in ev.read(*source)[0, 0]]
            fvars[vals["identifier"]] = ("float4", value)
        if node.type in INPUT_TYPES and not top:
            source = inputs.get(vals["identifier"])
            if source is not None:  # the instance's connection passes straight through
                results[node_id] = {"output": source}
                continue
        if node.type == "output" and not top:
            if "input" in srcs:
                outputs[vals["identifier"]] = srcs["input"]
            continue

        defaults = PARENT_DEFAULTS if node.type in INPUT_TYPES else NODE_DEFAULTS
        extra = d.extra(vals, package) if node.type != INSTANCE else {}
        base = {**defaults, **d.base_defaults_for(vals, extra), **node.base}
        main = srcs.get(d.primary_input()) if d.inputs else None
        # The node's functions share their variables: base parameters first, then the parameters top to
        # bottom; what they Set reaches its FX-Map.
        scope = NodeScope(set_variables(node.function_graphs()))
        own = _resolve(_base_functions(graph, node.id, base, ctx, main, fvars, scope), ctx, main)
        if node.functions:
            _apply_functions(graph, node, d, vals, own, fvars, scope)
        if node.type == INSTANCE:
            sub_pkg, sub_graph = package.resolve(node.ref)
            sub_inputs = {port: srcs.get(port) for port in d.inputs}
            sub_ctx = _resolve({**PARENT_DEFAULTS, **sub_graph.base}, own, None)
            _, sub_out = _compile(sub_pkg, sub_graph, vals, sub_inputs, sub_ctx, path + "/", jobs, top=False, ev=ev)
            results[node_id] = sub_out
            continue

        input_types = {port: (srcs[port][0].output_types[srcs[port][1]] if port in srcs else None)
                       for port in d.inputs}
        if any(p.kind in (ParamKind.FUNCTION, ParamKind.FXMAP) for p in d.params.values()):
            # Function parameters (Pixel processor…) may Get the graph's parameters and input values, an FX-Map
            # what the node's functions Set.
            extra = {**extra, "graph_params": {k: [kind, v] for k, (kind, v) in fvars.items()}}
            if scope.values:
                extra["node_variables"] = scope.values
        out_types = d.output_types(input_types, vals)
        value_only = bool(out_types) and all(t is PortType.VALUE for t in out_types.values())
        size = (1, 1) if value_only else tuple(1 << max(0, s - own.shift) for s in own.size)  # values: 1x1
        job = Job("", path, d, vals, {port: srcs.get(port) for port in d.inputs}, out_types, size,
                  {o: ("32f" if out_types[o] is PortType.VALUE else own.depth) for o in d.outputs},
                  TILINGS[own.tiling], float(own.seed),
                  extra={**extra, "pixel_size": own.pixel_size, "pixel_ratio": own.pixel_ratio, "context": own},
                  partition=PREVIEW if own.shift and not value_only else "")
        job.key = _key(job)
        jobs[path] = job
        results[node_id] = {o: (job, o) for o in d.outputs}
        if node.type == "output":
            outputs[vals["identifier"]] = (job, "output")
    return results, outputs


def system_variables(ctx: Context) -> dict[str, Any]:
    """The system variables for a node with these base parameters."""
    return {"$size": [float(1 << ctx.size[0]), float(1 << ctx.size[1])], "$sizelog2": [float(s) for s in ctx.size],
            "$tiling": TILINGS[ctx.tiling], "$pixelratio": PIXEL_RATIOS.index(ctx.pixel_ratio), "$time": 0.0,
            "$normalformat": 1, "$physicalsize": [0.0, 0.0, 0.0]}


class NodeScope:
    """The variables a node's functions Set (`values`, as they are evaluated) and all those they can Set (`names`)."""

    def __init__(self, names: frozenset):
        self.names, self.values = names, {}

    def evaluate(self, fg, variables: dict, seed: float):
        """Evaluate `fg` over `variables` and what the node's functions have Set so far, keeping what it Sets."""
        from trafilatura3d.functions import evaluate
        env = {**variables, **self.values}
        value = evaluate(fg, env, seed=seed, shared=True)
        self.values.update({k: v for k, v in env.items() if k in self.names})
        return value


def _apply_functions(graph: Graph, node, d, vals: dict, ctx: Context, fvars: dict, scope: NodeScope) -> None:
    """Replace the values of parameters driven by functions with what the functions compute, top to bottom."""
    from trafilatura3d.functions import FunctionError, FunctionGraph, to_param
    variables = {**system_variables(ctx), **{k: v for k, (_, v) in fvars.items()}}
    for name in sorted(node.functions, key=lambda k: list(d.params).index(k) if k in d.params else len(d.params)):
        try:
            fg = FunctionGraph(node.functions[name], graph.function_variables(), scope.names)
            vals[name] = to_param(scope.evaluate(fg, variables, ctx.seed), d.params[name])
        except (FunctionError, KeyError, ZeroDivisionError, OverflowError, ValueError) as e:
            raise DocumentError(f"{graph.id}.{node.id}.{name}: function: {e}") from None


def _base_functions(graph: Graph, node_id: str, base: dict, parent: Context, main: tuple[Job, str] | None,
                    fvars: dict, scope: NodeScope) -> dict:
    """The base parameters with their functions computed: each sees the system variables of what its mode inherits
    from (the primary input, else the parent) and the graph's parameters."""
    from trafilatura3d.functions import PARAM_TYPES, FunctionError, FunctionGraph, to_param
    if not any("function" in b for b in base.values()):
        return base
    out = dict(base)
    for name, b in base.items():
        if "function" not in b:
            continue
        ref = main[0].extra["context"] if b["mode"] == RELATIVE_INPUT and main is not None else parent
        kind = BASE_PARAMS[name].kind
        try:
            fg = FunctionGraph(b["function"], graph.function_variables(), scope.names)
            if fg.types[fg.output] != PARAM_TYPES[kind]:
                raise FunctionError(f"gives a {fg.types[fg.output]}, {name} takes a {PARAM_TYPES[kind]}")
            value = scope.evaluate(fg, {**system_variables(ref), **{k: v for k, (_, v) in fvars.items()}}, ref.seed)
            value = to_param(value, Param(kind, None, name))
            out[name] = BASE_PARAMS[name].validate({"mode": b["mode"], "value": value})
        except (FunctionError, KeyError, ZeroDivisionError, OverflowError, ValueError) as e:
            raise DocumentError(f"{graph.id}.{node_id}: base parameter {name}: function: {e}") from None
    return out


def _resolve(base: dict, parent: Context, main: tuple[Job, str] | None) -> Context:
    """Concrete base parameters from inheritance settings, the parent and the primary input."""
    if main is not None:
        job, out = main
        src = job.extra["context"]
        inherited = Context(src.size, job.depths[out], src.pixel_size, src.pixel_ratio, src.tiling, src.seed,
                            src.shift)
    else:
        inherited = parent

    def pick(name):
        mode, value = base[name]["mode"], base[name]["value"]
        ref = getattr(parent if mode == RELATIVE_PARENT else inherited, _FIELD[name])
        return mode, value, ref

    mode, value, ref = pick("output_size")
    size = tuple(value) if mode == ABSOLUTE else (ref[0] + value[0], ref[1] + value[1])
    size = tuple(max(0, min(MAX_LOG2, SIZE_LIMIT_LOG2, s)) for s in size)  # Preferences: Cooking size limit
    mode, value, ref = pick("pixel_size")
    pixel_size = tuple(value) if mode == ABSOLUTE else (ref[0] * value[0], ref[1] * value[1])
    mode, value, ref = pick("seed")
    seed = value if mode == ABSOLUTE else ref + value
    plain = {}
    for name in ("output_format", "pixel_ratio", "tiling"):
        mode, value, ref = pick(name)
        plain[name] = value if mode == ABSOLUTE else ref
    return Context(size, plain["output_format"], pixel_size, plain["pixel_ratio"], plain["tiling"], seed,
                   parent.shift)


_FIELD = {"output_size": "size", "output_format": "depth", "pixel_size": "pixel_size",
          "pixel_ratio": "pixel_ratio", "tiling": "tiling", "seed": "seed"}


def _key(job: Job) -> str:
    extra = {k: v for k, v in job.extra.items() if k != "context"}
    data = [job.node.type, getattr(job.node, "pipeline_key", ""), job.values, job.size, job.depths, job.tiling,
            job.seed, {p: (s[0].key, s[1]) if s else None for p, s in job.inputs.items()}, extra]
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
