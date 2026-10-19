"""Designer's function graphs (<dynamicValue>) and FX-Map graphs (<paramsGraph>) as ours (trafilatura3d.functions,
trafilatura3d.fxmap). What has no counterpart raises `Unsupported`, which the importer reports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from trafilatura3d.fxmap import BLENDING, FX_PARAMS, FX_VARIABLES, PATTERNS
from trafilatura3d.functions import FNODES, PARAM_TYPES, FunctionError, FunctionGraph
from trafilatura3d.nodes.base import ParamKind
from trafilatura3d.sbs.reader import constant, parameters, text


class Unsupported(ValueError):
    pass


_TYPE_NAMES = {"float1": "float", "integer1": "int", "integer2": "int2", "integer3": "int3", "integer4": "int4",
               "int1": "int"}
# Designer operator -> (our node, {Designer port: ours}).
_OPS = {
    "add": ("add", {"a": "a", "b": "b"}), "sub": ("subtraction", {"a": "a", "b": "b"}),
    "mul": ("multiplication", {"a": "a", "b": "b"}), "div": ("division", {"a": "a", "b": "b"}),
    "mod": ("modulo", {"a": "a", "b": "b"}), "dot": ("dot_product", {"a": "a", "b": "b"}),
    "mulscalar": ("scalar_multiplication", {"a": "a", "scalar": "scalar"}), "neg": ("negation", {"a": "a"}),
    "and": ("and", {"a": "a", "b": "b"}), "or": ("or", {"a": "a", "b": "b"}), "not": ("not", {"a": "a"}),
    "eq": ("equal", {"a": "a", "b": "b"}), "noteq": ("not_equal", {"a": "a", "b": "b"}),
    "gt": ("greater", {"a": "a", "b": "b"}), "gteq": ("greater_or_equal", {"a": "a", "b": "b"}),
    "lr": ("lower", {"a": "a", "b": "b"}), "lreq": ("lower_or_equal", {"a": "a", "b": "b"}),
    **{f: (f, {"a": "x"}) for f in ("pow2", "sqrt", "exp", "log", "log2", "abs", "ceil", "floor", "cos", "sin", "tan")},
    "rand": ("random", {"a": "x"}), "pow": ("pow", {"a": "x", "b": "y"}),
    "lerp": ("lerp", {"a": "a", "b": "b", "x": "x"}), "min": ("min", {"a": "a", "b": "b"}),
    "max": ("max", {"a": "a", "b": "b"}), "atan2": ("atan2", {"a": "vector"}),
    "cartesian": ("cartesian", {"rho": "length", "theta": "angle"}),
    "ifelse": ("if_else", {"condition": "condition", "ifpath": "if_", "elsepath": "else_"}),
    "sequence": ("sequence", {"seqin": "in_", "seqlast": "last"}), "set": ("set", {"value": "input"}),
    "samplelum": ("sample_gray", {"pos": "position"}), "samplecol": ("sample_color", {"pos": "position"}),
    **{f"vector{n}": (f"vector_float{n}", {"componentsin": "input_1", "componentslast": "input_2"}) for n in (2, 3, 4)},
    **{f"ivector{n}": (f"vector_int{n}", {"componentsin": "input_1", "componentslast": "input_2"}) for n in (2, 3, 4)},
    **{f"swizzle{n}": ("swizzle_float" + ("" if n == 1 else str(n)), {"vector": "vector"}) for n in (1, 2, 3, 4)},
    **{f"iswizzle{n}": ("swizzle_int" + ("" if n == 1 else str(n)), {"vector": "vector"}) for n in (1, 2, 3, 4)},
    **{f"tofloat{s}": (f"to_float{s}", {"value": "input"}) for s in ("", "2", "3", "4")},
    "toint1": ("to_integer", {"value": "input"}), **{f"toint{n}": (f"to_integer{n}", {"value": "input"}) for n in (2, 3, 4)},
    "while": ("while_loop", {"init": "init", "cond": "exit_cond", "loop": "loop_body"}),
    "passthrough": ("passthrough", {"input": "input"}), "isdefined": ("is_defined", {}),
}


def function(dyn: ET.Element, names: dict[str, str] | None = None, calls=None) -> dict:
    """A <dynamicValue> as our function graph. `names`: Designer variable -> ours (renamed graph inputs). `calls(path)`
    says what a call of the function graph at `path` (pkg:///...?dependency=uid) becomes: (our Library node type, "",
    {}) or ("call", the function's name, the functions table it needs); without it calls are Unsupported."""
    names = names or {}
    nodes, referenced, functions = [], set(), {}
    for pn in dyn.iter("paramNode"):
        uid, fn = text(pn, "uid"), text(pn, "function")
        data = constant(pn.find("funcDatas"))
        conns = {text(c, "identifier"): text(c, "connRef") for c in pn.findall("connections/connection")}
        referenced.update(conns.values())
        node: dict[str, Any] = {"id": f"n{uid}", "type": "", "params": {}, "inputs": {}, "position": [0.0, 0.0]}
        pos = text(pn, "GUILayout/gpos")
        if pos:
            node["position"] = [float(x) for x in pos.split()[:2]]
        if fn.startswith(("const_", "get_")):
            kind, t = fn.split("_", 1)
            node["type"] = f"{kind}_{_TYPE_NAMES.get(t, t)}"
            if node["type"] not in FNODES:
                raise Unsupported(f"function node {fn!r}")
            if kind == "const":
                # Designer leaves a constant at zero unwritten: zero of the constant's own type (a vector's too)
                value = data if data is not None else FNODES[node["type"]].params["value"].default
                if t in ("int3", "int4", "integer3", "integer4") and data is not None:
                    value = " ".join(str(int(x)) for x in value)
                node["params"]["value"] = value
            else:
                node["params"]["variable"] = names.get(str(data), str(data))
        elif fn in _OPS:
            node["type"], ports = _OPS[fn]
            node["inputs"] = {ports[p]: f"n{src}" for p, src in conns.items() if p in ports}
            if fn in ("set", "isdefined"):
                node["params"]["variable"] = names.get(str(data), str(data))
            elif fn == "while" and data is not None:
                node["params"]["max_iterations"] = int(data)
            elif fn.startswith(("swizzle", "iswizzle")):
                comps = data if isinstance(data, list) else [data or 0]
                node["params"] = {"xyzw"[i]: "XYZW"[int(c)] for i, c in enumerate(comps)}
            elif fn in ("samplelum", "samplecol"):
                index, filtering = (list(data) + [0, 0])[:2] if isinstance(data, list) else (data or 0, 0)
                node["params"] = {"input_image": int(index),
                                  "filtering_mode": ("Bilinear filtering", "Nearest")[int(filtering)] if int(filtering) in (0, 1)
                                  else "Bilinear filtering"}
        elif fn == "instance":
            if calls is None:
                raise Unsupported(f"calls the function graph {str(data).split('?')[0].replace('pkg:///', '')}")
            node["type"], name, table = calls(str(data))
            ports = set(table[name]["inputs"]) if name else {p for p, _ in FNODES[node["type"]].inputs}
            node["inputs"] = {p: f"n{src}" for p, src in conns.items() if p in ports}
            if name:
                node["params"]["function"] = name
            functions.update(table)
        else:
            raise Unsupported(f"function node {fn!r}")
        if not node["params"]:
            del node["params"]
        if not node["inputs"]:
            del node["inputs"]
        nodes.append(node)
    root = text(dyn, "rootnode")
    if not root:  # an old file without a root: the node nothing reads
        free = [n["id"] for n in nodes if n["id"][1:] not in referenced]
        root = free[-1][1:] if free else ""
    if not nodes:
        raise Unsupported("empty function")
    return {"nodes": nodes, "output": f"n{root}", **({"functions": functions} if functions else {})}


def adapt(f: dict, want: str, variables: dict[str, str], color: bool = False, outer: frozenset = frozenset()) -> dict:
    """The function made to give `want`, as Designer converts a function's value to its parameter: an integer and a
    float of the same size into each other, a float spread to a float4 (a grayscale value: alpha 1 for a `color`), a
    float3 given alpha 1; raises Unsupported otherwise (FunctionError when it doesn't type with `variables`)."""
    have = FunctionGraph(f, variables, outer).types[f["output"]]
    if have == want:
        return f
    base = lambda t: t.rstrip("234")
    size = lambda t: t[-1] if t[-1].isdigit() else ""
    nodes, out = list(f["nodes"]), f["output"]

    def add(id, type, inputs=None, params=None):
        nodes.append({"id": id, "type": type, "inputs": inputs or {}, "params": params or {}, "position": [0.0, 0.0]})
        return id
    if size(have) == size(want) and {base(have), base(want)} == {"int", "float"}:
        out = add("cast_out", f"to_float{size(have)}" if base(want) == "float" else f"to_integer{size(have)}",
                  {"input": out})
    elif have == "bool" and want == "int":  # a condition as 1 / 0 (FX-Map Switch selectors)
        out = add("bool_int", "if_else", {"condition": out, "if_": add("one", "const_int", params={"value": 1}),
                                          "else_": add("zero", "const_int", params={"value": 0})})
    elif have == "float" and want == "float4":
        two = add("spread_xy", "vector_float2", {"input_1": out, "input_2": out})
        if color:
            three = add("spread_xyz", "vector_float3", {"input_1": two, "input_2": out})
            out = add("spread_xyzw", "vector_float4", {"input_1": three, "input_2": add("alpha", "const_float",
                                                                                         params={"value": 1.0})})
        else:
            out = add("spread_xyzw", "vector_float4", {"input_1": two, "input_2": two})
    elif have == "float3" and want == "float4":
        out = add("with_alpha", "vector_float4", {"input_1": out, "input_2": add("alpha", "const_float",
                                                                                  params={"value": 1.0})})
    else:
        raise Unsupported(f"gives a {have}, the parameter takes a {want}")
    return {**f, "nodes": nodes, "output": out}


def scaled(f: dict, factor: float) -> dict:
    """A float function multiplied by `factor`."""
    nodes = list(f["nodes"]) + [
        {"id": "scale_k", "type": "const_float", "params": {"value": factor}, "position": [0.0, 0.0]},
        {"id": "scale_out", "type": "multiplication", "inputs": {"a": f["output"], "b": "scale_k"},
         "position": [0.0, 0.0]}]
    return {**f, "nodes": nodes, "output": "scale_out"}


def remap(f: dict, values: dict[int, int]) -> dict:
    """An integer function remapped through `values` (Designer's enumeration value -> our choice index):
    if out == a then a' else if ... else out."""
    if all(a == b for a, b in values.items()):
        return f
    nodes = list(f["nodes"])
    result = f["output"]
    for i, (a, b) in enumerate(values.items()):
        if a == b:
            continue
        nodes += [{"id": f"remap{i}_a", "type": "const_int", "params": {"value": a}, "position": [0.0, 0.0]},
                  {"id": f"remap{i}_b", "type": "const_int", "params": {"value": b}, "position": [0.0, 0.0]},
                  {"id": f"remap{i}_eq", "type": "equal", "inputs": {"a": f["output"], "b": f"remap{i}_a"},
                   "position": [0.0, 0.0]},
                  {"id": f"remap{i}", "type": "if_else", "inputs": {"condition": f"remap{i}_eq", "if_": f"remap{i}_b",
                                                                    "else_": result}, "position": [0.0, 0.0]}]
        result = f"remap{i}"
    return {**f, "nodes": nodes, "output": result}


def single_get(f: dict) -> str | None:
    """The variable a function only reads (Designer's way of exposing a parameter), else None."""
    if len(f["nodes"]) == 1 and f["nodes"][0]["type"].startswith("get_"):
        return f["nodes"][0]["params"]["variable"]
    return None


# FX-Map: Designer's node types, their child slots (<connection> output0..3) and parameters.
_FX_TYPES = {"paramset": ("quadrant", ("q1", "q2", "q3", "q4")), "addnode": ("iterate", ("pass_through", "repeated")),
             "markov2": ("switch", ("input_1", "input_2"))}
_FX_PARAMS = {
    "quadrant": {"patterntype": "pattern", "patternsize": "pattern_size", "frameoffset": "pattern_offset",
                 "branchoffset": "branch_offset", "opacity": "color_luminosity", "patternrotation": "pattern_rotation",
                 "patternsuppl": "pattern_variation", "blendingmode": "blending_mode", "randomseed": "random_seed",
                 "randominherited": "inherit_random", "imageindex": "input_image",
                 "imagefiltering": "input_image_filtering"},
    "iterate": {"numberadded": "iterations", "randomseed": "random_seed"},
    "switch": {"switch": "selector"},
}
# Quadrant enumerations: Designer's value -> our choice. patterntype is the index in PATTERNS. blendingmode: 0 Add/Sub,
# 1 Max, 2 Alpha blend, our order.
_FX_ENUMS = {"pattern": dict(enumerate(PATTERNS)), "blending_mode": dict(enumerate(BLENDING)),
             "input_image_filtering": {0: "Bilinear", 1: "Nearest"}}


def fxmap(pg: ET.Element, names: dict[str, str], variables: dict[str, str], problems: list[str], calls=None,
          outer: frozenset = frozenset()) -> dict:
    """A <paramsGraph> as our FX-Map graph; what can't be converted is appended to `problems` (and left at our
    default). `variables`: the graph's parameters functions may Get, with their types; `calls`: as `function`'s;
    `outer`: what the FX-Map node's functions (its own and the tree's) Set."""
    datas = {text(d, "uid"): d for d in pg.iter("paramsGraphData")}
    nodes = []
    for gn in pg.iter("paramsGraphNode"):
        uid, kind = text(gn, "uid"), text(gn, "type")
        if kind not in _FX_TYPES:
            problems.append(f"FX-Map node {kind!r} not converted")
            continue
        ours, slots = _FX_TYPES[kind]
        node: dict[str, Any] = {"id": f"fx{uid}", "type": ours, "params": {}, "functions": {}, "children": {},
                                "position": [float(x) for x in (text(gn, "GUILayout/gpos", "0 0").split()[:2])]}
        for c in gn.findall("connections/connection"):
            i = int(text(c, "identifier").replace("output", "") or 0)
            if i < len(slots):
                node["children"][slots[i]] = f"fx{text(c, 'connRef')}"
        data = datas.get(text(gn, "data"))
        for name, value in parameters(data.find("parameters") if data is not None else None).items():
            key = _FX_PARAMS[ours].get(name)
            if key is None:
                problems.append(f"FX-Map {ours} parameter {name!r} not converted")
                continue
            param, enum = FX_PARAMS[ours][key], _FX_ENUMS.get(key)
            try:
                if value.dynamic is not None:
                    f = adapt(function(value.dynamic, names, calls), PARAM_TYPES[param.kind], {**FX_VARIABLES, **variables},
                              param.kind is ParamKind.COLOR, outer)
                    if enum is not None:
                        f = remap(f, {a: list(param.choices).index(c) for a, c in enum.items()})
                    FunctionGraph(f, {**FX_VARIABLES, **variables}, outer)  # raises on what our functions can't do
                    node["functions"][key] = f
                else:
                    v = value.constant
                    if enum is not None:
                        v = enum[int(v)]
                    elif key == "color_luminosity" and not isinstance(v, list):
                        v = [float(v)] * 3 + [1.0]
                    elif param.kind is ParamKind.BOOL:
                        v = bool(v)
                    node["params"][key] = param.validate(v)
            except (Unsupported, FunctionError, KeyError, ValueError) as e:
                problems.append(f"FX-Map {ours} {key}: {e}")
        nodes.append(node)
    return {"nodes": nodes, "root": f"fx{text(pg, 'rootnode')}"}
