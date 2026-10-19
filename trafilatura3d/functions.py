"""Function graphs: small graphs of typed value nodes (constants, vectors, swizzles,
variables, samplers, casts, operators, logic, comparisons, maths, control flow) that compute a
value. They drive node parameters (evaluated on the CPU at compile time) and the Pixel processor
(compiled to WGSL and run per pixel).

A function graph is {"nodes": [{"id", "type", "params", "inputs": {port: node id}, "position"}],
"output": node id, "functions": {name: function}}. A Call node runs one of the `functions` (each a function graph
with "inputs": {port: type} and optional "defaults": {port: value}; one table for the whole call tree) on the values
of its inputs, and every node of the Library category (LIBRARY: the library functions, written here as our
own function graphs) is a call of a built-in one. Variables are shared through calls: a function
reads the caller's variables and what it Sets is seen after the call; only its inputs are its own.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from trafilatura3d.nodes.base import Param, ParamKind

TYPES = ("float", "float2", "float3", "float4", "int", "int2", "int3", "int4", "bool", "string")
WGSL_TYPE = {"float": "f32", "float2": "vec2f", "float3": "vec3f", "float4": "vec4f", "int": "i32",
             "int2": "vec2i", "int3": "vec3i", "int4": "vec4i", "bool": "bool"}
COMPONENTS = "xyzw"


class FunctionError(ValueError):
    pass


def width(t: str) -> int:
    return int(t[-1]) if t[-1].isdigit() else 1


def base(t: str) -> str:
    return t.rstrip("234")


def with_width(b: str, n: int) -> str:
    return b if n == 1 else f"{b}{n}"


# --- node definitions ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FNode:
    type: str
    label: str
    category: str
    inputs: tuple[tuple[str, str], ...] = ()  # (port, label)
    params: dict[str, Param] = field(default_factory=dict)


FNODES: dict[str, FNode] = {}
LIBRARY: dict[str, dict] = {}  # Library node type -> the function graph it calls (written at the end of the module)


def _add(type, label, category, inputs=(), params=None):
    FNODES[type] = FNode(type, label, category, tuple(inputs), params or {})


_CONST_DEFAULT = {"float": 0.0, "float2": [0.0, 0.0], "float3": [0.0, 0.0, 0.0], "float4": [0.0, 0.0, 0.0, 0.0],
                  "int": 0, "int2": [0, 0]}
for _t, _label in [("float", "Float"), ("float2", "Float2"), ("float3", "Float3"), ("float4", "Float4"),
                   ("int", "Integer"), ("int2", "Integer2"), ("int3", "Integer3"), ("int4", "Integer4"),
                   ("bool", "Boolean"), ("string", "String")]:
    _kind = {"float": ParamKind.FLOAT, "float2": ParamKind.FLOAT2, "float3": ParamKind.FLOAT3,
             "float4": ParamKind.FLOAT4, "int": ParamKind.INT, "int2": ParamKind.INT2, "bool": ParamKind.BOOL,
             "string": ParamKind.STRING}.get(_t)
    _default = {"bool": False, "string": "", "int3": "0 0 0", "int4": "0 0 0 0"}.get(_t, _CONST_DEFAULT.get(_t))
    if _kind is None:  # int3/int4: typed as text "x y z", the parameter kinds have no int3/int4
        _kind = ParamKind.STRING
    _add(f"const_{_t}", _label, "Constant", params={"value": Param(_kind, _default, "Value")})
    _add(f"get_{_t}", f"Get {_label}", "Variables", params={"variable": Param(ParamKind.STRING, "", f"Get {_t}")})
for _t in ("float2", "float3", "float4", "int2", "int3", "int4"):
    _add(f"vector_{_t}", f"Vector {'Float' if _t.startswith('float') else 'Integer'}{_t[-1]}", "Vector",
         [("input_1", "Input 1"), ("input_2", "Input 2")])
for _t in ("float", "float2", "float3", "float4", "int", "int2", "int3", "int4"):
    _n = width(_t)
    _add(f"swizzle_{_t}", f"Swizzle {'Float' if _t.startswith('float') else 'Integer'}{_t[-1] if _n > 1 else ''}",
         "Swizzle", [("vector", "Vector")],
         {COMPONENTS[i]: Param(ParamKind.ENUM, "XYZW"[i], f"Component {COMPONENTS[i].upper()}", choices=tuple("XYZW"))
          for i in range(_n)})
    if _t.startswith("float"):
        _add(f"to_{_t.replace('float', 'integer')}", f"To Integer{_t[5:]}", "Cast", [("input", "Input")])
    else:
        _add(f"to_{_t.replace('int', 'float')}", f"To Float{_t[3:]}", "Cast", [("input", "Input")])
_add("set", "Set", "Variables", [("input", "Input")], {"variable": Param(ParamKind.STRING, "", "Set")})
_add("is_defined", "Is defined", "Variables", params={"variable": Param(ParamKind.STRING, "", "Variable")})
_SAMPLER = {"input_image": Param(ParamKind.INT, 0, "Input image", 0, 7),
            "filtering_mode": Param(ParamKind.ENUM, "Bilinear filtering", "Filtering mode",
                                    choices=("Nearest", "Bilinear filtering")),
            "addressing_mode": Param(ParamKind.ENUM, "Repeat", "Addressing mode", choices=("Repeat", "Clamp to Edge"))}
_add("sample_gray", "Sample Gray", "Samplers", [("position", "Position")], dict(_SAMPLER))
_add("sample_color", "Sample Color", "Samplers", [("position", "Position")], dict(_SAMPLER))
for _t, _label in [("add", "Add"), ("subtraction", "Subtraction"), ("multiplication", "Multiplication"),
                   ("division", "Division"), ("modulo", "Modulo"), ("dot_product", "Dot Product")]:
    _add(_t, _label, "Operator", [("a", "a"), ("b", "b")])
_add("scalar_multiplication", "Scalar Multiplication", "Operator", [("a", "a"), ("scalar", "scalar")])
_add("negation", "Negation", "Operator", [("a", "a")])
for _t, _label in [("and", "And"), ("or", "Or")]:
    _add(_t, _label, "Logical", [("a", "a"), ("b", "b")])
_add("not", "Not", "Logical", [("a", "a")])
for _t, _label in [("equal", "Equal"), ("not_equal", "Not Equal"), ("greater", "Greater"),
                   ("greater_or_equal", "Greater or Equal"), ("lower", "Lower"), ("lower_or_equal", "Lower or Equal")]:
    _add(_t, _label, "Comparison", [("a", "a"), ("b", "b")])
for _t, _label in [("pow2", "2Pow"), ("sqrt", "Square root"), ("exp", "Exponential"), ("log", "Logarithm"),
                   ("log2", "Logarithm base 2"), ("abs", "Absolute"), ("ceil", "Ceil"), ("floor", "Floor"),
                   ("cos", "Cosine"), ("sin", "Sine"), ("tan", "Tangent"), ("random", "Random")]:
    _add(_t, _label, "Function", [("x", "X")])
_add("pow", "Pow", "Function", [("x", "X"), ("y", "Y")])
_add("lerp", "Linear interpolation", "Function", [("a", "A"), ("b", "B"), ("x", "X")])
_add("min", "Minimum", "Function", [("a", "A"), ("b", "B")])
_add("max", "Maximum", "Function", [("a", "A"), ("b", "B")])
_add("atan2", "Arc tangent 2", "Function", [("vector", "Vector")])
_add("cartesian", "Cartesian", "Function", [("length", "Length"), ("angle", "Angle")])
_add("if_else", "If...Else", "Control", [("condition", "Condition"), ("if_", "If"), ("else_", "Else")])
_add("sequence", "Sequence", "Control", [("in_", "In"), ("last", "Last")])
_add("while_loop", "While Loop", "Control", [("init", "Init."), ("exit_cond", "Exit Cond."), ("loop_body", "Loop Body")],
     {"max_iterations": Param(ParamKind.INT, 64, "Max. iterations", -1, 1024)})
_add("passthrough", "Passthrough", "Control", [("input", "Input")])
# Its ports are the called function's inputs.
_add("call", "Function call", "Control", params={"function": Param(ParamKind.STRING, "", "Function")})

_ARITH = {"add", "subtraction", "multiplication", "division", "modulo"}
_COMPARE = {"equal": "==", "not_equal": "!=", "greater": ">", "greater_or_equal": ">=", "lower": "<",
            "lower_or_equal": "<="}
_WGSL_ARITH = {"add": "+", "subtraction": "-", "multiplication": "*", "division": "/", "modulo": "%"}
_FLOAT_FN = {"pow2", "sqrt", "exp", "log", "log2", "abs", "ceil", "floor", "cos", "sin", "tan"}


# --- graph ----------------------------------------------------------------------------------------

class FunctionGraph:
    """A validated function graph with inferred types. `variables`: name -> type of the variables
    Get nodes may read (built-ins of the context and graph parameters); `outer`: variables Set elsewhere it may Get
    too (by the functions calling it), typed by its Gets. The functions it calls are typed with it (`callee`);
    `_functions` and `_calling` are given to those: the call tree's table and the functions being typed above (a
    function calling itself is refused)."""

    def __init__(self, data: dict, variables: dict[str, str], outer: frozenset = frozenset(),
                 _functions: dict | None = None, _calling: tuple = ()):
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
            raise FunctionError("a function graph is {'nodes': [...], 'output': id}")
        self.data = data
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.output = data.get("output")
        self.variables = dict(variables)
        self.types: dict[str, str] = {}
        self.functions = data.get("functions", {}) if _functions is None else _functions
        self.outer, self._calling = frozenset(outer), _calling
        self._callees: dict[tuple, FunctionGraph] = {}
        for n in self.nodes.values():
            if n["type"] not in FNODES:
                raise FunctionError(f"unknown function node {n['type']!r}")
        if self.output not in self.nodes:
            raise FunctionError("the function graph has no output node")
        self._sets = {self.params(k)["variable"]: k for k, n in self.nodes.items() if n["type"] == "set"}
        self.set_names = _set_names(data, self.functions)  # also what the functions it calls Set
        for name in self._sets:
            if name in data.get("inputs", {}):
                raise FunctionError(f"Set {name!r}: a function can't Set its own input")
        self.type_of(self.output)
        # Every Set and Get of a variable agree on its type, through the calls too.
        uses: dict[str, set[str]] = {}
        for k, n in self.nodes.items():
            if k in self.types and (n["type"] == "set" or n["type"].startswith("get_")):
                name = self.params(k)["variable"]
                if name not in self.variables:
                    uses.setdefault(name, set()).add(self.types[k])
        for c in self._callees.values():
            for name, types in c.var_uses.items():
                if name not in self.variables:
                    uses.setdefault(name, set()).update(types)
        for name, types in uses.items():
            if len(types) > 1:
                raise FunctionError(f"variable {name!r} used as {' and '.join(sorted(types))}")
        self.var_uses = uses  # variable -> type, for those Set or read without being built-ins or parameters

    def params(self, node_id: str) -> dict[str, Any]:
        n = self.nodes[node_id]
        return {k: n.get("params", {}).get(k, p.default) for k, p in FNODES[n["type"]].params.items()}

    def input(self, node_id: str, port: str) -> str:
        src = self.nodes[node_id].get("inputs", {}).get(port)
        if src is None or src not in self.nodes:
            raise FunctionError(f"{node_id}: input {port!r} is not connected")
        return src

    def connected(self, node_id: str, port: str) -> bool:
        return self.nodes[node_id].get("inputs", {}).get(port) in self.nodes

    def callee(self, node_id: str) -> FunctionGraph:
        """The function a Call or Library node runs, typed with the variables of this one and its own inputs."""
        key, body, functions = _callee(self.nodes[node_id], self.params(node_id), self.functions)
        if body is None:
            raise FunctionError(f"{node_id}: unknown function {key[1]!r}")
        if key not in self._callees:
            if key in self._calling:
                raise FunctionError(f"function {key[1]!r} calls itself")
            self._callees[key] = FunctionGraph(body, {**self.variables, **body.get("inputs", {})},
                                               self.outer | self.set_names, functions, self._calling + (key,))
        return self._callees[key]

    def type_of(self, node_id: str, _stack=()) -> str:
        if node_id in self.types:
            return self.types[node_id]
        if node_id in _stack:
            raise FunctionError(f"cycle through {node_id!r}")
        stack = _stack + (node_id,)
        t = self.nodes[node_id]["type"]
        p = self.params(node_id)
        ins = lambda port: self.type_of(self.input(node_id, port), stack)
        if t.startswith("const_") or t.startswith("get_"):
            out = t.split("_", 1)[1]
            if t.startswith("const_"):  # a value of its own type: the interpreter and WGSL take it as it is
                try:
                    FNODES[t].params["value"].validate(p["value"])
                except ValueError as e:
                    raise FunctionError(f"{node_id}: {e}") from None
            if t.startswith("get_"):
                # A Get declares its type; Set variables are checked against it once everything is typed
                # (a loop reads and writes the same variable, so inferring through the Set would cycle).
                name = p["variable"]
                if name in self.variables:
                    if self.variables[name] != out:
                        raise FunctionError(f"Get {out}: {name!r} holds a {self.variables[name]}")
                elif name not in self.set_names and name not in self.outer:
                    raise FunctionError(f"Get: unknown variable {name!r}")
        elif t.startswith("vector_"):
            out = t.split("_", 1)[1]
            a, b = ins("input_1"), ins("input_2")
            if base(a) != base(out) or base(b) != base(out) or width(a) + width(b) != width(out):
                raise FunctionError(f"{node_id}: {a} and {b} don't make a {out}")
        elif t.startswith("swizzle_"):
            out = t.split("_", 1)[1]
            v = ins("vector")
            if base(v) != base(out):
                raise FunctionError(f"{node_id}: swizzle of {v} into {out}")
            for comp in COMPONENTS[:width(out)]:  # a scalar is all its components
                if width(v) > 1 and "XYZW".index(p[comp]) >= width(v):
                    raise FunctionError(f"{node_id}: {v} has no component {p[comp]}")
        elif t.startswith("to_"):
            v = ins("input")
            want = "float" if t.startswith("to_float") else "int"
            out = with_width(want, width(v))  # a value of that type already passes through
            if base(v) not in ("float", "int"):
                raise FunctionError(f"{node_id}: can't cast {v}")
        elif t == "set":
            out = ins("input")
        elif t == "is_defined":
            out = "bool"
        elif t in ("sample_gray", "sample_color"):
            if ins("position") != "float2":
                raise FunctionError(f"{node_id}: Position must be a float2")
            out = "float" if t == "sample_gray" else "float4"
        elif t in _ARITH:  # an unconnected operand is the zero of the other's type
            a = ins("a") if self.connected(node_id, "a") or not self.connected(node_id, "b") else ins("b")
            b = ins("b") if self.connected(node_id, "b") else a
            if a != b or base(a) not in ("float", "int"):
                raise FunctionError(f"{node_id}: {a} and {b} must be the same numeric type")
            out = a
        elif t == "scalar_multiplication":
            a, s = ins("a"), ins("scalar")
            if base(a) != "float" or s != "float":
                raise FunctionError(f"{node_id}: a float vector times a float")
            out = a
        elif t == "negation":
            out = ins("a")
            if base(out) not in ("float", "int"):
                raise FunctionError(f"{node_id}: can't negate {out}")
        elif t == "dot_product":
            a, b = ins("a"), ins("b")
            if a != b or base(a) != "float":
                raise FunctionError(f"{node_id}: dot of two same float vectors")
            out = "float"
        elif t in ("and", "or", "not"):
            for port in (["a"] if t == "not" else ["a", "b"]):
                if ins(port) != "bool":
                    raise FunctionError(f"{node_id}: logic needs booleans")
            out = "bool"
        elif t in _COMPARE:  # an integer and a float compare as floats
            if ins("a") not in ("float", "int") or ins("b") not in ("float", "int"):
                raise FunctionError(f"{node_id}: compare two scalars")
            out = "bool"
        elif t in _FLOAT_FN:  # component-wise on float vectors
            x = ins("x")
            if x not in ("float", "int") and base(x) != "float":
                raise FunctionError(f"{node_id}: needs a scalar or a float vector")
            out = with_width("float", width(x))
        elif t == "random":
            if ins("x") not in ("float", "int"):
                raise FunctionError(f"{node_id}: needs a scalar")
            out = "float"
        elif t == "pow":
            ins("x"), ins("y")
            out = "float"
        elif t == "lerp":
            a, b = ins("a"), ins("b")
            if a != b or base(a) not in ("float", "int") or ins("x") != "float":
                raise FunctionError(f"{node_id}: lerp of two same values by a float")
            out = with_width("float", width(a))
        elif t in ("min", "max"):
            a, b = ins("a"), ins("b")
            if a != b or base(a) not in ("float", "int"):
                raise FunctionError(f"{node_id}: {t} of two same numeric values")
            out = a
        elif t == "atan2":
            if ins("vector") != "float2":
                raise FunctionError(f"{node_id}: needs a float2")
            out = "float"
        elif t == "cartesian":
            if ins("length") != "float" or ins("angle") != "float":
                raise FunctionError(f"{node_id}: needs two floats")
            out = "float2"
        elif t == "if_else":
            if ins("condition") != "bool":
                raise FunctionError(f"{node_id}: the condition must be a boolean")
            a, b = ins("if_"), ins("else_")
            if a != b:
                raise FunctionError(f"{node_id}: If and Else must have the same type")
            out = a
        elif t == "sequence":  # In, Init. and Exit Cond. are optional
            if self.connected(node_id, "in_"):
                ins("in_")
            out = ins("last")
        elif t == "while_loop":
            if self.connected(node_id, "init"):
                ins("init")
            if self.connected(node_id, "exit_cond") and ins("exit_cond") != "bool":
                raise FunctionError(f"{node_id}: Exit Cond. must be a boolean")
            if not self.connected(node_id, "exit_cond") and p["max_iterations"] < 0:
                raise FunctionError(f"{node_id}: without Exit Cond. nor Max. iterations the loop never ends")
            out = ins("loop_body")
        elif t == "passthrough":
            out = ins("input")
        elif t == "call" or t in LIBRARY:
            callee = self.callee(node_id)
            inputs = callee.data.get("inputs", {})
            for port, src in self.nodes[node_id].get("inputs", {}).items():
                if port not in inputs:
                    raise FunctionError(f"{node_id}: the function has no input {port!r}")
                if self.type_of(self.input(node_id, port), stack) != inputs[port]:
                    raise FunctionError(f"{node_id}: input {port!r} takes a {inputs[port]}")
            out = callee.types[callee.output]
        else:
            raise FunctionError(f"unsupported function node {t!r}")
        self.types[node_id] = out
        return out

    @classmethod
    def infer_types(cls, data: dict, variables: dict[str, str]) -> dict[str, str]:
        """Types of every node that can be typed, for a graph that may be incomplete or wrong
        (the editor shows them while the graph is being built)."""
        fg = cls.__new__(cls)
        fg.data, fg.variables, fg.types = data, dict(variables), {}
        fg.nodes = {n["id"]: n for n in data.get("nodes", []) if n.get("type") in FNODES}
        fg.output = data.get("output")
        fg.functions, fg.outer, fg._calling, fg._callees = data.get("functions", {}), frozenset(), (), {}
        fg._sets = {fg.params(k)["variable"]: k for k, n in fg.nodes.items() if n["type"] == "set"}
        fg.set_names = _set_names(data, fg.functions)
        for k in fg.nodes:
            try:
                fg.type_of(k)
            except (FunctionError, KeyError):
                pass
        return fg.types


def _callee(node: dict, params: dict, functions: dict) -> tuple[tuple, dict | None, dict]:
    """(key, body, the table its own calls resolve in) of the function a Call or Library node runs."""
    if node["type"] == "call":
        return ("call", params["function"]), functions.get(params["function"]), functions
    return ("library", node["type"]), LIBRARY[node["type"]], {}


def _set_names(data: dict, functions: dict, _seen: frozenset = frozenset()) -> frozenset:
    """The variables a function graph Sets, itself or in the functions it calls."""
    names = set()
    for n in data.get("nodes", []):
        t = n.get("type")
        if t == "set":
            names.add(n.get("params", {}).get("variable", ""))
        elif t == "call" or t in LIBRARY:
            key, body, table = _callee(n, {"function": n.get("params", {}).get("function", "")}, functions)
            if body is not None and key not in _seen:
                names |= _set_names(body, table, _seen | {key})
    return frozenset(names)


def set_variables(functions) -> frozenset:
    """What function graphs Set, with the functions they call: what the other functions of the same compositing node
    may Get: a node's functions are evaluated in one scope (base parameters first, then the parameters top to
    bottom, then its FX-Map)."""
    return frozenset().union(*(_set_names(f, f.get("functions", {})) for f in functions))


def zero(value_type: str):
    """The zero of a value type (what an unconnected function input without a default reads)."""
    n = width(value_type)
    z = {"float": 0.0, "int": 0, "bool": False, "string": ""}[base(value_type)]
    return z if n == 1 or value_type == "string" else [z] * n


class _Scope:
    """The variables inside a called function: its inputs, then the caller's (written through: what the function
    Sets is the caller's)."""

    def __init__(self, local: dict, outer):
        self.local, self.outer = local, outer

    def __getitem__(self, name):
        return self.local[name] if name in self.local else self.outer[name]

    def __setitem__(self, name, value):
        (self.local if name in self.local else self.outer)[name] = value

    def __contains__(self, name):
        return name in self.local or name in self.outer

    def get(self, name, default=None):
        return self[name] if name in self else default


def call_arguments(fg: FunctionGraph, node_id: str, value) -> dict:
    """The input values of a call: `value(port)` for the connected ones, the defaults (or zeros) for the others."""
    callee = fg.callee(node_id)
    connected = fg.nodes[node_id].get("inputs", {})
    defaults = callee.data.get("defaults", {})
    return {port: value(port) if port in connected else defaults.get(port, zero(t))
            for port, t in callee.data.get("inputs", {}).items()}


# --- CPU interpreter (parameter functions, value processor without samplers) ----------------------

def _vec(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _op(f, a, b):
    if isinstance(a, (list, tuple)):
        return [f(x, y) for x, y in zip(a, b)]
    return f(a, b)


def _int3(text):
    return [int(x) for x in str(text).replace(",", " ").split()]


def evaluate(fg: FunctionGraph, variables: dict[str, Any], seed: float = 0.0, sampler=None, shared: bool = False,
             _path: str = ""):
    """Evaluate the graph's output. `variables`: Get values (with `shared`, Set nodes write into this
    dict, so later functions of the same node see them, parameters being evaluated top to
    bottom); `sampler(index, gray, pos, filtering, addressing)` answers Sample nodes (None: they raise).
    `_path`: the calls this graph runs in (two calls of a function draw different random numbers)."""
    env = variables if shared else dict(variables)

    def rand(node_id):
        key = f"{seed}:{_path}{node_id}:{env.get('$number', 0)}:{env.get('$pos')}:{env.get('$depth')}"
        return int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "little") / 2**32

    def ev(k, memo):
        """Node k's value; each node is computed once (`memo`), as in the WGSL: a node shared by
        two branches doesn't run its Sets twice. Loop bodies and If...Else branches compute their nodes again."""
        if k not in memo:
            memo[k] = compute(k, memo)
        return memo[k]

    def compute(k, memo):
        n = fg.nodes[k]
        t = n["type"]
        p = fg.params(k)
        inp = lambda port: ev(fg.input(k, port), memo)
        out_t = fg.types[k]
        if t.startswith("const_"):
            v = p["value"]
            return _int3(v) if out_t in ("int3", "int4") else v
        if t.startswith("get_"):
            return env[p["variable"]]
        if t.startswith("vector_"):
            return _vec(inp("input_1")) + _vec(inp("input_2"))
        if t.startswith("swizzle_"):
            v = _vec(inp("vector"))
            r = [v["XYZW".index(p[c])] if len(v) > 1 else v[0] for c in COMPONENTS[:width(out_t)]]
            return r[0] if len(r) == 1 else r
        if t.startswith("to_float"):
            v = inp("input")
            return [float(x) for x in v] if isinstance(v, list) else float(v)
        if t.startswith("to_integer"):
            v = inp("input")
            rnd = lambda x: int(math.floor(x + 0.5))
            return [rnd(x) for x in v] if isinstance(v, list) else rnd(v)
        if t == "set":
            v = inp("input")
            env[p["variable"]] = v
            return v
        if t == "is_defined":
            return p["variable"] in env
        if t in ("sample_gray", "sample_color"):
            if sampler is None:
                raise FunctionError("samplers are not available here")
            return sampler(p["input_image"], t == "sample_gray", inp("position"), p["filtering_mode"],
                           p["addressing_mode"])
        if t in _ARITH:
            a, b = (inp(port) if fg.connected(k, port) else zero(out_t) for port in ("a", "b"))
            integer = base(out_t) == "int"
            f = {"add": lambda x, y: x + y, "subtraction": lambda x, y: x - y, "multiplication": lambda x, y: x * y,
                 "division": (lambda x, y: int(x / y) if y else 0) if integer else (lambda x, y: x / y if y else 0.0),
                 "modulo": (lambda x, y: x % y if y else 0) if integer else (lambda x, y: math.fmod(x, y) if y else 0.0)}[t]
            return _op(f, a, b)
        if t == "scalar_multiplication":
            a, s = inp("a"), inp("scalar")
            return [x * s for x in a] if isinstance(a, list) else a * s
        if t == "negation":
            a = inp("a")
            return [-x for x in a] if isinstance(a, list) else -a
        if t == "dot_product":
            return float(sum(x * y for x, y in zip(_vec(inp("a")), _vec(inp("b")))))
        if t == "and":
            return bool(inp("a")) and bool(inp("b"))
        if t == "or":
            return bool(inp("a")) or bool(inp("b"))
        if t == "not":
            return not inp("a")
        if t in _COMPARE:
            a, b = inp("a"), inp("b")
            return {"equal": a == b, "not_equal": a != b, "greater": a > b, "greater_or_equal": a >= b,
                    "lower": a < b, "lower_or_equal": a <= b}[t]
        if t in _FLOAT_FN:
            x = inp("x")
            f = {"pow2": lambda v: 2.0 ** v, "sqrt": lambda v: math.sqrt(max(v, 0.0)), "exp": math.exp,
                 "log": lambda v: math.log(v) if v > 0 else -math.inf, "log2": lambda v: math.log2(v) if v > 0 else -math.inf,
                 "abs": abs, "ceil": math.ceil, "floor": math.floor, "cos": math.cos, "sin": math.sin, "tan": math.tan}[t]
            return [float(f(float(v))) for v in x] if isinstance(x, list) else float(f(float(x)))
        if t == "random":
            return rand(k) * float(inp("x"))
        if t == "pow":
            return float(float(inp("x")) ** float(inp("y")))
        if t == "lerp":
            a, b, x = inp("a"), inp("b"), inp("x")
            return _op(lambda u, v: (1 - x) * u + x * v, a, b)
        if t in ("min", "max"):
            return _op(min if t == "min" else max, inp("a"), inp("b"))
        if t == "atan2":
            v = inp("vector")
            return math.atan2(v[1], v[0])
        if t == "cartesian":
            r, a = inp("length"), inp("angle")
            return [r * math.cos(a), r * math.sin(a)]
        if t == "if_else":
            return ev(fg.input(k, "if_" if inp("condition") else "else_"), dict(memo))
        if t == "sequence":
            if fg.connected(k, "in_"):
                inp("in_")
            return inp("last")
        if t == "while_loop":
            if fg.connected(k, "init"):
                inp("init")
            limit = p["max_iterations"]
            body, i = None, 0
            exits = (lambda: ev(fg.input(k, "exit_cond"), {})) if fg.connected(k, "exit_cond") else (lambda: False)
            while (limit < 0 or i < limit) and not exits():
                body = ev(fg.input(k, "loop_body"), {})
                i += 1
            return body if body is not None else ev(fg.input(k, "loop_body"), {})
        if t == "passthrough":
            return inp("input")
        if t == "call" or t in LIBRARY:
            args = call_arguments(fg, k, inp)
            return evaluate(fg.callee(k), _Scope(args, env), seed, sampler, True, f"{_path}{k}/")
        raise FunctionError(f"unsupported function node {t!r}")

    return ev(fg.output, {})


# --- WGSL code generation (Pixel processor) -------------------------------------------------------

class WGSLCodegen:
    """Statements computing the graph's output into a WGSL variable. Get variables map to WGSL
    expressions given in `names`; samplers call `sample_<port>`/`texel_<port>` of the node's inputs
    (`input_ports[i]` for Input image i); Random hashes the pixel and the node id. A call is written inline: the
    function's statements with its inputs bound to the arguments (`_locals`), in the caller's variables (`_root`
    declares them all and numbers the temporaries)."""

    def __init__(self, fg: FunctionGraph, names: dict[str, str], input_ports: list[str], _root=None,
                 _locals: dict[str, str] | None = None, _path: str = ""):
        self.fg, self.names, self.ports = fg, names, input_ports
        self.root, self.locals, self.path = _root or self, _locals or {}, _path
        if _root is None:
            self.lines: list[str] = []
            self.counter = 0
            self.decls = [f"var v_{_ident(name)}: {WGSL_TYPE[next(iter(types))]};"
                          for name, types in fg.var_uses.items() if next(iter(types)) in WGSL_TYPE]

    def tmp(self) -> str:
        self.root.counter += 1
        return f"t{self.root.counter}"

    def emit(self) -> tuple[str, str]:
        """(statements, result expression)"""
        r = self.expr(self.fg.output, {}, self.lines, "    ")
        return "\n".join(self.decls + self.lines), r

    def expr(self, k, memo, out, ind) -> str:
        if k in memo:
            return memo[k]
        fg = self.fg
        n = fg.nodes[k]
        t = n["type"]
        p = fg.params(k)
        ot = fg.types[k]
        sub = lambda port: self.expr(fg.input(k, port), memo, out, ind)
        if t == "if_else":
            c = sub("condition")
            r = self.tmp()
            out.append(f"{ind}var {r}: {WGSL_TYPE[ot]};")
            out.append(f"{ind}if ({c}) {{")
            a = self.expr(fg.input(k, "if_"), dict(memo), out, ind + "    ")
            out.append(f"{ind}    {r} = {a};")
            out.append(f"{ind}}} else {{")
            b = self.expr(fg.input(k, "else_"), dict(memo), out, ind + "    ")
            out.append(f"{ind}    {r} = {b};")
            out.append(f"{ind}}}")
            memo[k] = r
            return r
        if t == "sequence":
            if fg.connected(k, "in_"):
                sub("in_")
            memo[k] = sub("last")
            return memo[k]
        if t == "call" or t in LIBRARY:
            callee = fg.callee(k)
            args = call_arguments(fg, k, sub)
            args = {port: (v if isinstance(v, str) else _literal(v, callee.data["inputs"][port]))
                    for port, v in args.items()}
            inner = WGSLCodegen(callee, self.names, self.ports, self.root, args, f"{self.path}{k}/")
            memo[k] = inner.expr(callee.output, {}, out, ind)
            return memo[k]
        if t == "passthrough":
            memo[k] = sub("input")
            return memo[k]
        if t == "while_loop":
            if fg.connected(k, "init"):
                sub("init")
            r = self.tmp()
            limit = p["max_iterations"]
            out.append(f"{ind}var {r}: {WGSL_TYPE[ot]};")
            out.append(f"{ind}for (var it_{r} = 0; {'true' if limit < 0 else f'it_{r} < {limit}'}; it_{r}++) {{")
            if fg.connected(k, "exit_cond"):
                cond = self.expr(fg.input(k, "exit_cond"), {}, out, ind + "    ")
                out.append(f"{ind}    if ({cond}) {{ break; }}")
            body = self.expr(fg.input(k, "loop_body"), {}, out, ind + "    ")
            out.append(f"{ind}    {r} = {body};")
            out.append(f"{ind}}}")
            memo[k] = r
            return r
        e = self._pure(k, t, p, ot, sub)
        r = self.tmp()
        if t == "set":
            out.append(f"{ind}v_{_ident(p['variable'])} = {e};")
        out.append(f"{ind}let {r} = {e};")
        memo[k] = r
        return r

    def _pure(self, k, t, p, ot, sub) -> str:
        T = WGSL_TYPE.get(ot, "f32")
        if t.startswith("const_"):
            return _literal(_int3(p["value"]) if ot in ("int3", "int4") else p["value"], ot)
        if t.startswith("get_"):
            name = p["variable"]
            if name in self.locals:
                return self.locals[name]
            if name in self.fg.set_names or name in self.fg.outer:
                return f"v_{_ident(name)}"
            return f"bool({self.names[name]})" if ot == "bool" else self.names[name]  # boolean uniforms are i32
        if t.startswith("vector_"):
            return f"{T}({sub('input_1')}, {sub('input_2')})"
        if t.startswith("swizzle_"):
            if width(self.fg.types[self.fg.input(k, "vector")]) == 1:  # a scalar's X repeated (WGSL can't swizzle it)
                return f"{T}({sub('vector')})" if width(ot) > 1 else sub("vector")
            comps = "".join("xyzw"["XYZW".index(p[c])] for c in COMPONENTS[:width(ot)])
            return f"({sub('vector')}).{comps}"
        if t.startswith("to_float"):
            return f"{T}({sub('input')})"
        if t.startswith("to_integer"):
            if base(self.fg.types[self.fg.input(k, "input")]) == "int":
                return sub("input")
            return f"{T}(round({sub('input')}))"
        if t == "set":
            return sub("input")
        if t == "is_defined":
            name = p["variable"]
            known = name in self.locals or name in self.names or name in self.fg.set_names or name in self.fg.outer
            return "true" if known else "false"
        if t in ("sample_gray", "sample_color"):
            port = self.ports[min(p["input_image"], len(self.ports) - 1)] if self.ports else None
            if port is None:
                return "0.0" if t == "sample_gray" else "vec4f(0.0)"
            pos = sub("position")
            if p["addressing_mode"] == "Repeat":
                pos = f"fract({pos})"
            else:
                pos = f"clamp({pos}, vec2f(0.0), vec2f(1.0))"
            if p["filtering_mode"] == "Nearest":
                v = f"texel_{port}(vec2i(floor({pos} * vec2f(size_{port}()))))"
            else:
                v = f"sample_{port}({pos})"
            return f"luminance(({v}).rgb)" if t == "sample_gray" else v
        if t in _ARITH:
            a, b = (sub(port) if self.fg.connected(k, port) else _literal(zero(ot), ot) for port in ("a", "b"))
            if t == "modulo" and base(ot) == "int":  # never % on signed integers (nodes/int_math.wgsl)
                return f"imod{'' if width(ot) == 1 else f'_v{width(ot)}'}({a}, {b})"
            return f"({a} {_WGSL_ARITH[t]} {b})"
        if t == "scalar_multiplication":
            return f"({sub('a')} * {sub('scalar')})"
        if t == "negation":
            return f"(-{sub('a')})"
        if t == "dot_product":
            a, b = sub("a"), sub("b")
            return f"dot({a}, {b})" if width(self.fg.types[self.fg.input(k, 'a')]) > 1 else f"({a} * {b})"
        if t == "and":
            return f"({sub('a')} && {sub('b')})"
        if t == "or":
            return f"({sub('a')} || {sub('b')})"
        if t == "not":
            return f"(!{sub('a')})"
        if t in _COMPARE:
            a, b = sub("a"), sub("b")
            if self.fg.types[self.fg.input(k, "a")] != self.fg.types[self.fg.input(k, "b")]:
                a, b = f"f32({a})", f"f32({b})"
            return f"({a} {_COMPARE[t]} {b})"
        if t in _FLOAT_FN:
            x = sub("x") if base(self.fg.types[self.fg.input(k, "x")]) == "float" else f"f32({sub('x')})"
            return {"pow2": f"exp2({x})", "sqrt": f"sqrt(max({x}, {T}(0.0)))", "exp": f"exp({x})", "log": f"log({x})",
                    "log2": f"log2({x})", "abs": f"abs({x})", "ceil": f"ceil({x})", "floor": f"floor({x})",
                    "cos": f"cos({x})", "sin": f"sin({x})", "tan": f"tan({x})"}[t]
        if t == "random":
            salt = int(hashlib.sha1(f"{self.path}{k}".encode()).hexdigest()[:6], 16) % 100000
            return f"(f32({sub('x')}) * hash21(c, p.seed_ + {salt}.0))"
        if t == "pow":
            return f"pow(f32({sub('x')}), f32({sub('y')}))"
        if t == "lerp":
            a, b, x = sub("a"), sub("b"), sub("x")
            return f"mix({T}({a}), {T}({b}), {x})"
        if t in ("min", "max"):
            return f"{t}({sub('a')}, {sub('b')})"
        if t == "atan2":
            v = sub("vector")
            return f"atan2(({v}).y, ({v}).x)"
        if t == "cartesian":
            r, a = sub("length"), sub("angle")
            return f"(vec2f(cos({a}), sin({a})) * {r})"
        raise FunctionError(f"unsupported function node {t!r}")


def _ident(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lstrip("$")) or "v"


def _literal(v, t: str) -> str:
    """A value of type `t` as a WGSL expression."""
    if t == "bool":
        return "true" if v else "false"
    if isinstance(v, list):
        return f"{WGSL_TYPE[t]}({', '.join(_lit(x, base(t)) for x in v)})"
    return _lit(v, t)


def _lit(v, t) -> str:
    if t == "int":
        return f"{int(v)}"
    f = float(v)
    if math.isinf(f) or math.isnan(f):
        return "0.0"
    return f"{f!r}" if "e" in repr(f) or "." in repr(f) else f"{f}.0"


# Parameter kinds and the function value types that drive them.
PARAM_TYPES = {ParamKind.FLOAT: "float", ParamKind.FLOAT2: "float2", ParamKind.FLOAT3: "float3",
               ParamKind.FLOAT4: "float4", ParamKind.COLOR: "float4", ParamKind.INT: "int", ParamKind.INT2: "int2",
               ParamKind.BOOL: "bool", ParamKind.ENUM: "int", ParamKind.STRING: "string"}

# Built-in variables of parameter functions (the system variables) and their types.
BUILTINS = {"$size": "float2", "$sizelog2": "float2", "$tiling": "int", "$pixelratio": "int", "$time": "float",
            "$normalformat": "int", "$physicalsize": "float3"}
PIXEL_BUILTINS = {**BUILTINS, "$pos": "float2"}


def to_param(value, param: Param):
    """A function result as a value of the parameter it drives."""
    k = param.kind
    if k is ParamKind.ENUM:
        return param.choices[max(0, min(len(param.choices) - 1, int(value)))]
    if k is ParamKind.INT:
        return int(value)
    if k is ParamKind.BOOL:
        return bool(value)
    if k is ParamKind.INT2:
        return [int(x) for x in value]
    if k in (ParamKind.FLOAT2, ParamKind.FLOAT3, ParamKind.FLOAT4, ParamKind.COLOR):
        return [float(x) for x in value]
    if k is ParamKind.FLOAT:
        return float(value)
    return value


def from_param(value, param: Param):
    """A parameter value as the function value type (for Get of graph parameters)."""
    if param.kind is ParamKind.ENUM:
        return param.choices.index(value)
    return value


def constant(value_type: str, value) -> dict:
    """A one-node function graph returning a constant (what exposing a parameter starts from)."""
    return {"nodes": [{"id": "c", "type": f"const_{value_type}", "params": {"value": value}, "position": [0, 0]}],
            "output": "c"}


# --- library functions ----------------------------------------------------------------------------
# The library functions (Math, Random, Transforms, Ranges, colour values, switches...) as Library
# nodes: what each computes, written here as our own function graphs. They keep the .sbs
# identifiers of the inputs (the ports) and run inline like any call, on the CPU and in WGSL.

class _X:
    """A value of a library function being written: a node of its graph, with its type."""

    def __init__(self, body, node_id: str, value_type: str):
        self.body, self.id, self.t = body, node_id, value_type

    def _like(self, v) -> _X:
        """`v` as a value: a Python number becomes a constant of this value's type (repeated in a vector)."""
        if isinstance(v, _X):
            return v
        if isinstance(v, (list, tuple)):
            return self.body.const([float(x) for x in v])
        b = "int" if base(self.t) == "int" else "float"
        x = int(v) if b == "int" else float(v)
        return self.body.const(x if width(self.t) == 1 else [x] * width(self.t), with_width(b, width(self.t)))

    def _op(self, op, other, swap=False) -> _X:
        a, b = (self._like(other), self) if swap else (self, self._like(other))
        if op == "multiplication" and a.t != b.t:  # a vector times a float
            v, s = (a, b) if width(a.t) > 1 else (b, a)
            return self.body.node("scalar_multiplication", {"a": v, "scalar": s})
        return self.body.node(op, {"a": a, "b": b})

    def __add__(self, o): return self._op("add", o)
    def __radd__(self, o): return self._op("add", o, True)
    def __sub__(self, o): return self._op("subtraction", o)
    def __rsub__(self, o): return self._op("subtraction", o, True)
    def __mul__(self, o): return self._op("multiplication", o)
    def __rmul__(self, o): return self._op("multiplication", o, True)
    def __truediv__(self, o): return self._op("division", o)
    def __rtruediv__(self, o): return self._op("division", o, True)
    def __mod__(self, o): return self._op("modulo", o)
    def __neg__(self): return self.body.node("negation", {"a": self})
    def __lt__(self, o): return self._op("lower", o)
    def __le__(self, o): return self._op("lower_or_equal", o)
    def __gt__(self, o): return self._op("greater", o)
    def __ge__(self, o): return self._op("greater_or_equal", o)

    def __getattr__(self, name):  # swizzles: v.x, v.zwxy
        if not name or len(name) > 4 or any(ch not in COMPONENTS for ch in name):
            raise AttributeError(name)
        t = base(self.t) + ("" if len(name) == 1 else str(len(name)))
        return self.body.node(f"swizzle_{t}", {"vector": self}, **{COMPONENTS[i]: ch.upper() for i, ch in enumerate(name)})


class _Body:
    """The graph of a library function being written; nodes are typed as they are added."""

    def __init__(self, inputs: dict[str, str]):
        self.nodes: list[dict] = []
        fg = FunctionGraph.__new__(FunctionGraph)
        fg.data, fg.nodes, fg.types, fg.output = {"nodes": self.nodes, "inputs": inputs}, {}, {}, None
        fg.variables = {**PIXEL_BUILTINS, **inputs}
        fg.functions, fg.outer, fg._calling, fg._callees, fg._sets = {}, set(), (), {}, {}
        fg.set_names = set()
        self.fg = fg

    def node(self, node_type: str, inputs: dict | None = None, **params) -> _X:
        k = f"n{len(self.nodes)}"
        n: dict[str, Any] = {"id": k, "type": node_type}
        if params:
            n["params"] = params
        if inputs:
            n["inputs"] = {port: x.id for port, x in inputs.items()}
        if node_type == "set":
            self.fg.set_names.add(params["variable"])
        self.nodes.append(n)
        self.fg.nodes[k] = n
        return _X(self, k, self.fg.type_of(k))

    def const(self, v, value_type: str | None = None) -> _X:
        if value_type is None:
            value_type = "bool" if isinstance(v, bool) else with_width("float", len(v)) if isinstance(v, list) \
                else "int" if isinstance(v, int) else "float"
        return self.node(f"const_{value_type}", value=v)

    def get(self, name: str, value_type: str) -> _X:
        """A variable: a system one, or one the caller Sets (the non-square functions read them)."""
        if name not in self.fg.variables:
            self.fg.outer.add(name)
        return self.node(f"get_{value_type}", variable=name)


_B: _Body | None = None  # the body being written


def _lib(node_type: str, label: str, category: str, defaults: dict | None = None, **inputs: str):
    """Declare a library function: its body is the decorated Python function of its inputs (values)."""
    assert not node_type.startswith(("const_", "get_", "vector_", "swizzle_", "to_")), "a prefix of node families"

    def define(body):
        global _B
        _B = _Body(inputs)
        args = [_B.get(name, t) for name, t in inputs.items()]
        out = _lift(body(*args))
        LIBRARY[node_type] = {"inputs": dict(inputs), "nodes": _B.nodes, "output": out.id,
                              **({"defaults": defaults} if defaults else {})}
        FNODES[node_type] = FNode(node_type, label, f"Library/{category}",
                                  tuple((p, p.replace("_", " ")[:1].upper() + p.replace("_", " ")[1:]) for p in inputs))
        return body
    return define


def _lift(v) -> _X:
    return v if isinstance(v, _X) else _B.const(v)


def _fn(t: str, x) -> _X:
    return _B.node(t, {"x": _lift(x)})


def _join(*parts) -> _X:
    """A vector of floats or vectors, joined two by two (the Vector nodes)."""
    parts = [_lift(p) for p in parts]
    v = parts[0]
    for p in parts[1:]:
        v = _B.node(f"vector_{base(v.t)}{width(v.t) + width(p.t)}", {"input_1": v, "input_2": p})
    return v


def _if(c: _X, a, b) -> _X:
    a, b = _lift(a), _lift(b)
    a, b = (a, b._like(a) if isinstance(b, _X) and b.t != a.t else b) if isinstance(a, _X) else (a, b)
    return _B.node("if_else", {"condition": c, "if_": a, "else_": b if isinstance(b, _X) else a._like(b)})


def _eq(a: _X, b) -> _X:
    return a._op("equal", b)


def _ne(a: _X, b) -> _X:
    return a._op("not_equal", b)


def _and(a: _X, b: _X) -> _X:
    return _B.node("and", {"a": a, "b": b})


def _or(a: _X, b: _X) -> _X:
    return _B.node("or", {"a": a, "b": b})


def _not(a: _X) -> _X:
    return _B.node("not", {"a": a})


def _min(a: _X, b) -> _X:
    return _B.node("min", {"a": a, "b": a._like(b)})


def _max(a: _X, b) -> _X:
    return _B.node("max", {"a": a, "b": a._like(b)})


def _dot(a: _X, b) -> _X:
    return _B.node("dot_product", {"a": a, "b": a._like(b)})


def _pow(x: _X, y) -> _X:
    return _B.node("pow", {"x": x, "y": _lift(y)})


def _rand(x) -> _X:
    return _fn("random", x)


def _atan2(v: _X) -> _X:
    return _B.node("atan2", {"vector": v})


def _cartesian(length: _X, angle: _X) -> _X:
    return _B.node("cartesian", {"length": length, "angle": angle})


def _float(x: _X) -> _X:
    return _B.node(f"to_float{x.t[3:]}", {"input": x})


def _int(x: _X) -> _X:
    return _B.node(f"to_integer{x.t[5:]}", {"input": x})


def _set(name: str, x: _X) -> _X:
    return _B.node("set", {"input": x}, variable=name)


def _seq(*xs: _X) -> _X:
    v = xs[0]
    for x in xs[1:]:
        v = _B.node("sequence", {"in_": v, "last": x})
    return v


def _saturate(x: _X) -> _X:
    return _max(_min(x, 1.0), 0.0)


def _frac(x: _X) -> _X:
    return x - _fn("floor", x)


def _length(v: _X) -> _X:
    return _fn("sqrt", _dot(v, v))


def _hash11(p: _X) -> _X:  # "Hash without Sine" (see NOTICE)
    x = _frac(p * 0.1031)
    x = x * (x + 33.33)
    return _frac(x * (x + x))


def _turns(t: _X) -> _X:  # turns to radians
    return t * (2.0 * math.pi)


def _lib_pow(x: _X, n: _X) -> _X:
    """Pow: 0 for x <= 0, else 2^(log2(x) n)."""
    return _if(x <= 0.0, 0.0, _fn("pow2", _fn("log2", x) * n))


def _mx(rgb: _X) -> _X:
    return _max(_max(rgb.x, rgb.y), rgb.z)


def _mn(rgb: _X) -> _X:
    return _min(_min(rgb.x, rgb.y), rgb.z)


def _hue_hexagonal(rgb: _X) -> _X:
    r, g, b, mx = rgb.x, rgb.y, rgb.z, _mx(rgb)
    c = mx - _mn(rgb)
    h = _if(_eq(c, 0.0), 0.0, _if(_eq(r, mx), ((g - b) / c) % 6.0,
                                  _if(_eq(g, mx), (b - r) / c + 2.0, _if(_eq(b, mx), (r - g) / c + 4.0, 1.0))))
    return h / 6.0


def _hue_2_polar(rgb: _X) -> _X:
    r, g, b = rgb.x, rgb.y, rgb.z
    a = _atan2(_join((2.0 * r - g - b) * 0.5, (g - b) * (math.sqrt(3.0) * 0.5)))
    return (a / math.pi * 0.5 + 1.0) % 1.0


def _luma601(rgb: _X) -> _X:
    return _dot(rgb, [0.299, 0.587, 0.114])


def _srgb_to_linear(x: _X) -> _X:
    return _if(x <= 0.04045, x / 12.92, _pow((x + 0.055) / 1.055, 2.4))


def _linear_to_srgb(x: _X) -> _X:
    return _if(x <= 0.0031308, x * 12.92, 1.055 * _pow(x, 1.0 / 2.4) - 0.055)


def _matrix3(rows, v: _X) -> _X:
    return _join(*(_dot(v, row) for row in rows))


# Math
@_lib("degrees_to_radians", "Degrees to radians", "Math", deg="float")
def _(deg): return deg * (math.pi / 180.0)
@_lib("arc_sine", "Arc sine", "Math", x="float")
def _(x): return _atan2(_join(_fn("sqrt", 1.0 - x * x), x))
@_lib("arc_cosine", "Arc cosine", "Math", x="float")
def _(x): return _atan2(_join(x, _fn("sqrt", 1.0 - x * x)))
@_lib("even_count", "Even count", "Math", Input="float")
def _(i): return _fn("ceil", _fn("floor", i) / 2.0)
@_lib("odd_count", "Odd count", "Math", Input="float")
def _(i): return _fn("floor", _fn("floor", i) / 2.0)
@_lib("parity_test", "Parity test", "Math", Input="float")
def _(i):  # true for an even number: the integer part minus twice its half
    n = _fn("floor", i)
    return n - 2.0 * _fn("floor", n * 0.5) < 0.5
@_lib("ln", "Ln", "Math", a="float")
def _(a): return _fn("log", a) / math.e  # the natural logarithm divided by e
@_lib("pi", "Pi", "Math")
def _(): return math.pi
@_lib("two_pi", "2 Pi", "Math")
def _(): return 2.0 * math.pi
@_lib("library_pow", "Pow (Deprecated)", "Math", x="float", n="float")
def _(x, n): return _lib_pow(x, n)
@_lib("roughness", "Roughness", "Math", Roughness="float", CurrentLevel="int", TotalLevel="int", GlobalOpacity="float")
def _(r, level, total, opacity):  # fxmap.level_weights
    w = (1.0 - r) * _pow(r, _float(level)) / (1.0 - _pow(r, _float(total)))
    return opacity * _if(_eq(r, 0.0), 1.0, _if(_eq(r, 1.0), 1.0 / _float(total), w))
def _equal(a: _X, b: _X) -> _X:
    e = _eq(a.x, b.x)
    for c in COMPONENTS[1:width(a.t)]:
        e = _and(e, _eq(getattr(a, c), getattr(b, c)))
    return e


for _n in (2, 3, 4):
    @_lib(f"equality_float{_n}", f"Equality Float{_n}", "Math", A=f"float{_n}", B=f"float{_n}")
    def _(a, b): return _equal(a, b)
    @_lib(f"not_equal_float{_n}", f"Not equal Float{_n}", "Math", A=f"float{_n}", B=f"float{_n}")
    def _(a, b): return _not(_equal(a, b))
@_lib("equality_boolean", "Equality boolean", "Math", A="bool", B="bool")
def _(a, b): return _or(_and(a, b), _and(_not(a), _not(b)))
@_lib("not_equal_boolean", "Not equal boolean", "Math", A="bool", B="bool")
def _(a, b): return _not(_or(_and(a, b), _and(_not(a), _not(b))))
@_lib("curve", "Curve", "Math", curve="float", input="float")
def _(c, x):
    return _if(c < 0.0, -c * _fn("sqrt", 1.0 - (1.0 - x) * (1.0 - x)) + x * (1.0 + c),
               (1.0 - _fn("sqrt", 1.0 - x * x)) * c + (1.0 - c) * x)
@_lib("merge_float4", "Merge Float4", "Math", x="float", y="float", z="float", w="float")
def _(x, y, z, w): return _join(x, y, z, w)
@_lib("merge_float3", "Merge Float3", "Math", x="float", y="float", z="float")
def _(x, y, z): return _join(x, y, z)
for _n in (2, 3):
    @_lib(f"distance_float{_n}", f"Distance Float{_n}", "Math", a=f"float{_n}", b=f"float{_n}")
    def _(a, b): return _length(b - a)
    @_lib(f"length_float{_n}", f"Length Float{_n}", "Math", v=f"float{_n}")
    def _(v): return _length(v)
@_lib("sign", "Sign", "Math", x="float")
def _(x): return _if(x >= 0.0, 1.0, -1.0)
@_lib("reflect", "Reflect", "Math", i="float3", n="float3")
def _(i, n): return i - n * 2.0 * _dot(n, i)
@_lib("one_minus", "One minus", "Math", x="float")
def _(x): return 1.0 - x
@_lib("fmod", "Fmod", "Math", a="float", b="float")
def _(a, b):
    m = _frac(_fn("abs", a / b)) * _fn("abs", b)
    return _if(a < 0.0, -m, m)
@_lib("frac", "Frac", "Math", input="float")
def _(x): return _frac(x)
@_lib("cross_product", "Cross product", "Math", a="float3", b="float3")
def _(a, b): return a.yzx * b.zxy - a.zxy * b.yzx
@_lib("cross_product_float2", "Cross product Vec2", "Math", a="float2", b="float2")
def _(a, b): return a.x * b.y - a.y * b.x
@_lib("step", "Step", "Math", a="float", x="float")
def _(a, x): return _if(x >= a, 1.0, 0.0)
for _n in (2, 3, 4):
    @_lib(f"scalar_division_float{_n}", f"Scalar division Float{_n}", "Math", input=f"float{_n}", scalar="float")
    def _(v, s, n=_n): return v / _join(*[s] * n)
    @_lib(f"normalize_float{_n}", f"Normalize Vec{_n}", "Math", input=f"float{_n}")
    def _(v): return v * (1.0 / _length(v))
@_lib("clamp", "Clamp", "Math", input="float", min="float", max="float")
def _(x, lo, hi): return _max(_min(x, hi), lo)
@_lib("smoothstep", "Smoothstep", "Math", a="float", b="float", x="float")
def _(a, b, x):
    t = _saturate((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)
@_lib("saturate", "Saturate", "Math", input="float")
def _(x): return _saturate(x)
@_lib("saturate_float2", "Saturate Float2", "Math", input="float2")
def _(v): return _join(_saturate(v.x), _saturate(v.y))
for _n, _t in ((1, "float"), (2, "float2"), (3, "float3"), (4, "float4")):
    @_lib(f"average_float{_n}", f"Average Float{_n}", "Math", a=_t, b=_t)
    def _(a, b): return (a + b) * 0.5
@_lib("orthogonal_float2", "Orthogonal Vec2", "Math", input="float2")
def _(v): return v.yx * [1.0, -1.0]
@_lib("truncate_float1", "Truncate Float1", "Math", input="float", decimals="int")
def _(x, d):
    k = _lib_pow(_B.const(10.0), _float(d))
    return _if(d <= 0, x, _fn("floor", x * k) / k)
@_lib("round_float1", "Round Float1", "Math", input="float")
def _(x): return _fn("floor", x + 0.5)

# Random: Random gives [0, x[
@_lib("random_uniform_ab", "Random uniform [a, b[", "Random", A="float", B="float")
def _(a, b): return _rand(b - a) + a
@_lib("random_discrete_ab", "Random discrete [a, b]", "Random", A="float", B="float", Probability="float")
def _(a, b, p): return _if(_rand(1.0) >= p, a, b)
@_lib("random_uniform_signed", "Random uniform [-1, 1[", "Random")
def _(): return _rand(2.0) - 1.0
for _n, _a, _b in ((2, "A2", "B2"), (3, "A", "B"), (4, "A", "B")):
    @_lib(f"random_uniform_float{_n}", f"Random uniform Float{_n} [a,{' ' if _n == 4 else ''}b[", "Random", **{_a: f"float{_n}", _b: f"float{_n}"})
    def _(a, b, n=_n):
        d = b - a
        return _join(*(_rand(getattr(d, c)) for c in COMPONENTS[:n])) + a
@_lib("degrees_to_turns", "Degrees to turns", "Random", input="float")
def _(x): return x / 360.0
@_lib("global_random", "Global random", "Random", Input="float", Seed="int")
def _(x, seed):  # Input times a value in 0..1 fixed by the seed (the hash of hash11, defined below)
    return x * _hash11(_float(seed) + 0.5)
@_lib("normal_distribution", "Normal distribution", "Random", radius="float", uniform_random="float2")
def _(r, u):  # a 2D normal of deviation radius / 4: Rayleigh distance (inverse CDF of 1 - u, never 0), uniform angle
    return _cartesian(_fn("sqrt", -2.0 * _fn("log", 1.0 - _min(u.x, 0.9999))) * (r / 4.0), _turns(u.y))

# Transforms: 2x2 matrices are float4 (row by row), angles in turns
@_lib("directional_offset", "Directional offset", "Transforms", Direction="float", Distance="float")
def _(d, dist): return _join(_fn("cos", _turns(d)), _fn("sin", _turns(d))) * dist
@_lib("matrix_multiply", "Matrix multiply", "Transforms", MatrixA="float4", MatrixB="float4")
def _(a, b): return _join(a.x * b.x + a.y * b.z, a.x * b.y + a.y * b.w, a.z * b.x + a.w * b.z, a.z * b.y + a.w * b.w)
@_lib("rotation_matrix", "Rotation matrix", "Transforms", RotationW="float")
def _(w):
    c, s = _fn("cos", _turns(w)), _fn("sin", _turns(w))
    return _join(c, -s, s, c)
@_lib("scale_matrix", "Scale matrix", "Transforms", ScaleUV="float2")
def _(s): return _join(100.0 / s.x, 0.0, 0.0, 100.0 / s.y)
@_lib("tile_matrix", "Tile matrix", "Transforms", tiles_xy="float2")
def _(t): return _join(t.x, 0.0, 0.0, t.y)
@_lib("rotate_float2", "Rotate Vec2", "Transforms", vec2="float2", angle="float", pivot="float2")
def _(v, a, pivot):
    d, c, s = v - pivot, _fn("cos", _turns(a)), _fn("sin", _turns(a))
    return _join(_dot(d, _join(c, s)), _dot(d, _join(-s, c))) + pivot
@_lib("rotate_float2_radians", "Rotate Vec2 (Radian)", "Transforms", vec2="float2", angle="float")
def _(v, a):
    c, s = _fn("cos", a), _fn("sin", a)
    return _join(_dot(v, _join(c, s)), _dot(v, _join(-s, c)))
@_lib("matrix_invert", "Matrix invert", "Transforms", matrix22="float4")
def _(m): return m.wyzx * [1.0, -1.0, -1.0, 1.0] * (1.0 / (m.x * m.w - m.y * m.z))
@_lib("polar_to_cartesian", "Polar to cartesian", "Transforms", input="float2")
def _(p):
    lat = (p.y - 0.5) * math.pi
    return _join(_join(_fn("cos", _turns(p.x)), _fn("sin", _turns(p.x))) * _fn("cos", lat), -_fn("sin", lat))
@_lib("cartesian_to_polar", "Cartesian to polar", "Transforms", input="float3")
def _(v):
    u = (_atan2(v.xy) / (2.0 * math.pi)) % 1.0
    return _join(u, 1.0 - (0.5 + _atan2(_join(_length(v.xy), v.z)) / math.pi))

# Cycles, ranges, conversions
@_lib("wave", "Wave", "Cycles", Amplitude="float", Frequency="float", greaterThanZero="bool")
def _(a, f, positive):
    w = a * _fn("sin", _B.get("$time", "float") * f)
    return _if(positive, w + a, w)
@_lib("signed_to_unit", "[-1, 1] to [0,1]", "Ranges", Input="float")
def _(x): return (x + 1.0) / 2.0
@_lib("unit_to_signed", "[0, 1] to [-1, 1]", "Ranges", Input="float")
def _(x): return x * 2.0 - 1.0
@_lib("unit_reversed", "[0, 1] to [1, 0]", "Ranges", Input="float")
def _(x): return 1.0 - x
@_lib("negate_float1", "Negate Float1 (Deprecated)", "Ranges", Input="float")
def _(x): return -x
@_lib("height_balance", "Height balance", "Ranges", Depth_Balance="float")
def _(d): return _join(_max(d, 0.0), _min(1.0 + d, 1.0), _max(-d, 0.0), _min(1.0 - d, 1.0))
@_lib("unit_to_tent", "[0,1] to [0,1,0]", "Ranges", input="float")
def _(x): return 1.0 - _fn("abs", x * 2.0 - 1.0)
@_lib("range_to_unit", "[a,b] to [0,1]", "Ranges", X="float", A="float", B="float")
def _(x, a, b): return _min((_max(x, a) - a) / (b - a), 1.0)
@_lib("triangle_wave", "Triangle wave", "Ranges", input="float")
def _(x): return 1.0 - _fn("abs", x % 2.0 - 1.0)
@_lib("sawtooth_wave", "Sawtooth wave", "Ranges", x="float")
def _(x): return _if(_eq(x % 1.0, 0.0), _min(_fn("abs", x), 1.0), x % 1.0)
@_lib("boolean_to_float1", "Boolean to Float1", "Conversions", Boolean="bool")
def _(b): return _if(b, 1.0, 0.0)
@_lib("turns_to_degrees", "Turns to degrees", "Conversions", turns="float")
def _(t): return t * 360.0

# Colour values
@_lib("direction_to_normal", "Direction to normal", "Color", Direction="float", slopeAngle="float", yUp="bool")
def _(d, slope, y_up):
    s = _fn("sin", _turns(slope))
    x = s * _fn("cos", _turns(d) + math.pi)
    y = s * _fn("sin", _turns(d) + _if(y_up, 0.0, math.pi))
    return _join((x + 1.0) / 2.0, (y + 1.0) / 2.0, 1.0, 1.0)
@_lib("rgb_to_hsl", "RGB to HSL", "Color", RGB="float3")
def _(rgb):
    r, g, b, mx, mn = rgb.x, rgb.y, rgb.z, _mx(rgb), _mn(rgb)
    lum, d = (mx + mn) * 0.5, mx - mn
    h = _if(_eq(mx, r), (g - b) / d + _if(g < b, 6.0, 0.0), _if(_eq(mx, g), (b - r) / d + 2.0, (r - g) / d + 4.0))
    s = d / _if(lum > 0.5, 2.0 - mx - mn, mx + mn)
    return _if(_ne(mx, mn), _join(h / 6.0, s, lum), _join(0.0, 0.0, lum))
@_lib("hsl_to_rgb", "HSL to RGB", "Color", HSL="float3")
def _(hsl):
    lum = hsl.z
    q = lum + (0.5 - _fn("abs", 0.5 - lum)) * hsl.y
    p = lum * 2.0 - q
    f = _max(_min(_fn("abs", hsl.xxx - [0.5, 1.0 / 3.0, 2.0 / 3.0]) * [6.0, -6.0, -6.0] + [-1.0, 2.0, 2.0], 1.0), 0.0)
    return _join(p, p, p) + (_join(q, q, q) - _join(p, p, p)) * f
@_lib("random_color", "Random color", "Color", RGB="float3", RGB_Randomness="float3")
def _(rgb, k): return rgb + _join(*((_rand(1.0) * 2.0 - 1.0) * getattr(k, c) for c in "xyz"))
@_lib("random_luminosity", "Random luminosity", "Color", RGB="float3", Luminosity_Randomness="float")
def _(rgb, k):  # one random offset for the three channels: lighter or darker, same hue
    n = (_rand(1.0) * 2.0 - 1.0) * k
    return rgb + _join(n, n, n)
@_lib("rgb_lightness_average", "RGB lightness average", "Color", rgb="float3")
def _(rgb): return (rgb.x + rgb.y + rgb.z) / 3.0
@_lib("rgb_lightness_hexcone", "RGB lightness hexcone", "Color", rgb="float3")
def _(rgb): return _mx(rgb)
@_lib("rgb_lightness_bihexcone", "RGB lightness bi-hexcone", "Color", rgb="float3")
def _(rgb): return (_mx(rgb) + _mn(rgb)) * 0.5
@_lib("rgb_lightness_luma_rec601", "RGB lightness luma Rec. 601", "Color", rgb="float3")
def _(rgb): return _luma601(rgb)
@_lib("rgb_lightness_luma_rec709", "RGB lightness luma Rec. 709", "Color", rgb="float3")
def _(rgb): return _dot(rgb, [0.2126, 0.7152, 0.0722])
@_lib("rgb_chroma_hexagonal", "RGB chroma hexagonal", "Color", rgb="float3")
def _(rgb): return _mx(rgb) - _mn(rgb)
@_lib("rgb_hue_hexagonal", "RGB hue hexagonal", "Color", rgb="float3")
def _(rgb): return _hue_hexagonal(rgb)
@_lib("rgb_hue_2_polar", "RGB hue 2 polar", "Color", rgb="float3")
def _(rgb): return _hue_2_polar(rgb)
@_lib("rgb_saturation_hsv", "RGB saturation HSV", "Color", rgb="float3")
def _(rgb): return _if(_eq(_mx(rgb), 0.0), 0.0, (_mx(rgb) - _mn(rgb)) / _mx(rgb))
@_lib("rgb_saturation_hsl", "RGB saturation HSL", "Color", rgb="float3")
def _(rgb):
    lum = (_mx(rgb) + _mn(rgb)) * 0.5
    return _if(_eq(lum, 1.0), 0.0, (_mx(rgb) - _mn(rgb)) / (1.0 - _fn("abs", lum * 2.0 - 1.0)))
@_lib("rgb_saturation_hsi", "RGB saturation HSI", "Color", rgb="float3")
def _(rgb):
    avg = (rgb.x + rgb.y + rgb.z) / 3.0
    return _if(_eq(avg, 0.0), 0.0, 1.0 - _mn(rgb) / avg)
@_lib("rgb_to_hsv", "RGB to HSV", "Color", rgb="float3")
def _(rgb): return _join(_hue_hexagonal(rgb), _if(_eq(_mx(rgb), 0.0), 0.0, (_mx(rgb) - _mn(rgb)) / _mx(rgb)), _mx(rgb))
@_lib("rgb_to_hsi", "RGB to HSI", "Color", rgb="float3")
def _(rgb):
    avg = (rgb.x + rgb.y + rgb.z) / 3.0
    return _join(_hue_2_polar(rgb), _if(_eq(avg, 0.0), 0.0, 1.0 - _mn(rgb) / avg), avg)
@_lib("rgb_to_hcl", "RGB to HCL", "Color", rgb="float3")
def _(rgb): return _join(_hue_hexagonal(rgb), _mx(rgb) - _mn(rgb), _luma601(rgb))
def _hexagon(h: _X, c: _X, x: _X) -> _X:
    """The RGB of a hue's sixth of the hexagon (h = 6 hue), chroma c, second component x; black outside 0-6."""
    zero = _B.const(0.0)
    rgb = _B.const([0.0, 0.0, 0.0])
    for lo, (r, g, b) in reversed(list(enumerate(((c, x, zero), (x, c, zero), (zero, c, x), (zero, x, c),
                                                   (x, zero, c), (c, zero, x))))):
        inside = _and(h >= float(lo), h < 6.0 if lo == 5 else h <= float(lo + 1))
        rgb = _if(inside, _join(r, g, b), rgb)
    return rgb


def _tent6(h: _X) -> _X:
    return 1.0 - _fn("abs", h % 2.0 - 1.0)


@_lib("hsv_to_rgb", "HSV to RGB", "Color", hsv="float3")
def _(hsv):
    c, h = hsv.y * hsv.z, hsv.x * 6.0
    m = hsv.z - c
    return _join(m, m, m) + _hexagon(h, c, _tent6(h) * c)
@_lib("hsi_to_rgb", "HSI to RGB", "Color", hsi="float3")
def _(hsi):
    h, z = hsi.x * 6.0, _tent6(hsi.x * 6.0)
    c = hsi.y * (hsi.z * 3.0) / (z + 1.0)
    m = (1.0 - hsi.y) * hsi.z
    return _join(m, m, m) + _hexagon(h, c, z * c)
@_lib("hcl_to_rgb", "HCL to RGB", "Color", hcl="float3")
def _(hcl):
    h = hcl.x * 6.0
    rgb = _hexagon(h, hcl.y, _tent6(h) * hcl.y)
    m = hcl.z - _luma601(rgb)
    return _join(m, m, m) + rgb
@_lib("linear_to_srgb_luminance", "Linear to sRGB (Luminance)", "Color", input="float")
def _(x): return _linear_to_srgb(x)
@_lib("linear_to_srgb", "Linear to sRGB", "Color", input="float3")
def _(v): return _join(_linear_to_srgb(v.x), _linear_to_srgb(v.y), _linear_to_srgb(v.z))
@_lib("srgb_to_linear_luminance", "sRGB to linear (Luminance)", "Color", input="float")
def _(x): return _srgb_to_linear(x)
@_lib("srgb_to_linear", "sRGB to linear", "Color", input="float3")
def _(v): return _join(_srgb_to_linear(v.x), _srgb_to_linear(v.y), _srgb_to_linear(v.z))
@_lib("temperature_to_rgb", "Temperature to RGB", "Color", temperature="float")
def _(kelvin):
    # A blackbody in linear sRGB at luminance 1: Planck's law through the CIE 1931 colour matching functions (the
    # analytic fit of Wyman, Sloan and Shirley 2013) and the sRGB matrix, fitted per channel by a polynomial in
    # x = ln(T / 6500) over 1000-40000 K (error < 0.012); outside that range the ends are held.
    x = _fn("log", _min(_max(kelvin, 1000.0), 40000.0)) - math.log(6500.0)

    def poly(*c):
        v = _B.const(c[0])
        for k in c[1:]:
            v = v * x + k
        return v
    return _max(_join(poly(-0.00812956, 0.013007, 0.0439849, -0.175026, 0.364533, -0.518995, 1.0431),
                     poly(0.00324387, -0.00568844, -0.0195053, 0.0708601, -0.0943821, 0.052369, 0.983904),
                     poly(-0.0095055, 0.0189084, 0.0692983, -0.189871, -0.143597, 1.01139, 1.03374)), 0.0)
_ACESCG_TO_SRGB = ((1.7049, -0.62416, -0.0809141), (-0.129553, 1.13837, -0.00876801), (-0.0241261, -0.124633, 1.14882))
_SRGB_TO_ACESCG = ((0.613118, 0.341182, 0.0457873), (0.0699341, 0.918103, 0.0119328), (0.020463, 0.106769, 0.872716))
@_lib("acescg_to_linear_srgb", "ACEScg to linear sRGB", "Color", input="float3")
def _(v): return _matrix3(_ACESCG_TO_SRGB, v)
@_lib("linear_srgb_to_acescg", "Linear sRGB to ACEScg", "Color", input="float3")
def _(v): return _matrix3(_SRGB_TO_ACESCG, v)
@_lib("tonemap_aces", "ACES tonemapper", "Tonemappers", input="float3")
def _(v):  # Stephen Hill's fit of the RRT and ODT (MIT, as published in BakingLab's ACES.hlsl), between its matrices
    a = _matrix3(((0.59719, 0.35458, 0.04823), (0.076, 0.90834, 0.01566), (0.0284, 0.13383, 0.83777)), v)
    b = (a * (a + 0.0245786) - 0.000090537) / (a * (a * 0.983729 + 0.432951) + 0.238081)
    return _matrix3(((1.60475, -0.53108, -0.07367), (-0.10208, 1.10813, -0.00605), (-0.00327, -0.07276, 1.07602)), b)

# Various: the noises' disorder and non-square output
@_lib("disorder", "Disorder", "Various", disorder_radius="float", disorder_angle="float")
def _(radius, angle): return _cartesian(_rand(radius), _turns(_rand(2.0) - 1.0) + _turns(angle))
@_lib("disorder_v2", "Disorder v2", "Various", disorder_radius="float", disorder_angle="float")
def _(radius, angle): return _cartesian(radius * _rand(1.0), _turns(_rand(1.0) + angle))
@_lib("disorder_anim", "Disorder anim", "Various", disorder_angle="float", speed="float")
def _(angle, speed):  # a step of speed / 2 pi times a uniform factor in 0.25 .. 1, in a random direction
    return _cartesian(speed / math.tau * (1.0 - 0.75 * _rand(1.0)), _turns(angle + _rand(1.0)))


def _xy_sets(d: _X) -> list[_X]:
    return [_set("xy_difference", d), _set("xy_ratio", _join(1.0, 1.0 / _fn("pow2", d)))]


@_lib("non_square_output_size", "Non-square output size", "Various", scale="int", output_size="int2", non_square="bool")
def _(scale, size, non_square):
    """Sets the variables of a noise FX-Map in cells: render_region and tiling_region (the pattern's square of
    real_scale cells along the long side, centred, the short side's share of it), real_scale, xy_difference and
    xy_ratio (the image's aspect); gives its output size unchanged."""
    log2 = _B.get("$sizelog2", "float2")
    d = _if(non_square, log2.x - log2.y, 0.0)
    r = _fn("pow2", -_fn("abs", d))
    real = _int(_fn("pow2", _fn("abs", d))) * scale
    m = (_float(real) - 1.0) * 0.5
    cells = _join(-m, m, -m, m) + [0.0, 1.0, 0.0, 1.0]
    render = _join(0.0, 0.0, (1.0 - r) * 0.5, (1.0 - r) * 0.5) + _join(1.0, 1.0, r, r) * cells
    q = _float(scale) * ((_fn("pow2", _fn("abs", d)) - 1.0) * 0.5)
    tiling = cells + _join(0.0, 0.0, q, -q)
    wide = d > 0.0
    return _seq(_set("render_region", _if(wide, render, render.zwxy)),
                _set("tiling_region", _if(wide, tiling, tiling.zwxy)), _set("real_scale", real), *_xy_sets(d), size)


@_lib("non_square_output_size_v2", "Non-square output size v2", "Various", scale="float", output_size="int2",
      non_square="bool", tile_offset="float2")
def _(scale, size, non_square, offset):
    """The same in the v2 noises' frame: the region also shifted by Tile Offset (cells), real_scale the odd number
    of cells around the square, total_number the cells of the region's walk."""
    log2 = _B.get("$sizelog2", "float2")
    diff = log2.x - log2.y
    d = _if(non_square, diff, 0.0)
    p = _fn("pow2", _fn("abs", d))
    s = _max(scale, 1.0)
    m = (p * s - 1.0) * 0.5
    u = _join(-m, m) + [0.0, 1.0]
    r = _fn("pow2", -_fn("abs", d))
    cells = _join(u, u * _join(r, r) + (1.0 - _join(r, r)) * 0.5)
    dv = _join(diff, 0.0)
    stretch = _if(non_square, _fn("pow2", _fn("abs", _if(log2.x > log2.y, dv, dv.yx))), _B.const([1.0, 1.0]))
    region = _if(d > 0.0, cells, cells.zwxy) + (offset * [1.0, -1.0] * s).xxyy * stretch.xxyy
    odd = _fn("ceil", p) + (_fn("ceil", p) % 2.0 + 1.0)
    c = _fn("ceil", _fn("abs", region) - [-0.5, 0.5, -0.5, 0.5])
    n = _max(_max(c.x, c.y) * 2.0 + 1.0, _max(c.z, c.w) * 2.0 + 1.0)
    return _seq(_set("render_region", region), _set("tiling_region", region), _set("real_scale", _int(odd * odd)),
                _set("total_number", _int(n * n)), *_xy_sets(d), size)


@_lib("non_square_expansion_uv_scale", "Non-square expansion UV scale", "Various", position="float2",
      non_square_expansion="bool")
def _(pos, nse):
    log2 = _B.get("$sizelog2", "float2")
    v = _join(_max(log2.x, log2.y) - _min(log2.x, log2.y), 0.0)
    s = _fn("pow2", _if(log2.x > log2.y, v, v.yx))
    return _if(nse, s * pos - (s - [1.0, 1.0]) * [0.5, 0.5], pos)


@_lib("compute_total_number", "Compute total number", "Various", {"tile_multiplier": 1}, tile_multiplier="int")
def _(k):
    """The cells of the render region's walk, the region scaled by Tile Multiplier when the image doesn't tile
    (its default is 1)."""
    k = _if(_eq(_B.get("$tiling", "int"), 0), k, 1)
    c = _fn("ceil", _fn("abs", _B.get("render_region", "float4") * _float(k)) - [-0.5, 0.5, -0.5, 0.5])
    n = _max(_max(c.x, c.y) * 2.0 + 1.0, _max(c.z, c.w) * 2.0 + 1.0)
    return _seq(_set("c_tile_multiplier", _float(k)), _int(n * n))


# Hashes ("hash without sine": fractional parts of products), 3D vectors
@_lib("hash11", "Hash 1 to 1", "Hash", p="float")
def _(p): return _hash11(p)
@_lib("hash21", "Hash 2 to 1", "Hash", input="float2")
def _(p):
    q = _frac(p.xyx * 0.1031)
    q = q + _join(*[_dot(q, q.yzx + 33.33)] * 3)
    return _frac((q.x + q.y) * q.z)
@_lib("hash31", "Hash 3 to 1", "Hash", p="float3")
def _(p):
    q = _frac(p * 0.1031)
    q = q + _join(*[_dot(q, q.zyx + 31.32)] * 3)
    return _frac((q.x + q.y) * q.z)
@_lib("hash14", "Hash 1 to 4", "Hash", p="float")
def _(p):
    q = _frac(p.xxxx * [0.1031, 0.103, 0.0973, 0.1099])
    q = q + _join(*[_dot(q.wzxy + 33.33, q)] * 4)
    return _frac((q.xxyz + q.yzzw) * q.zywx)
@_lib("rotate_float3", "Vec3 rotation", "3D", input="float3", axis="float3", angle="float")
def _(v, axis, angle):  # Rodrigues, angle in turns
    c, s = _fn("cos", _turns(angle)), _fn("sin", _turns(angle))
    cross = axis.yzx * v.zxy - axis.zxy * v.yzx
    return v * c + cross * s + axis * _dot(axis, v) * (1.0 - c)
@_lib("matrix3_multiply_vector", "Mat3 multiply vector", "3D", mat3_0="float3", mat3_1="float3", mat3_2="float3",
      vec3="float3")
def _(m0, m1, m2, v): return m0 * v.x + m1 * v.y + m2 * v.z
@_lib("multiply_vector_matrix3", "Vec3 multiply mat3", "3D", vec3="float3", mat3_0="float3", mat3_1="float3",
      mat3_2="float3")
def _(v, m0, m1, m2): return _join(_dot(m0, v), _dot(m1, v), _dot(m2, v))


# Switches: Input Selection picks one of 2, 4 or 8 values (from 0; below 0 the first, above the last)
for _t, _name in (("float", "Float1"), ("float2", "Float2"), ("float3", "Float3"), ("float4", "Float4"),
                  ("int", "Integer1"), ("int2", "Integer2"), ("int3", "Integer3"), ("int4", "Integer4")):
    for _count in (2, 4, 8):
        @_lib(f"switch_{_name.lower()}_{_count}", f"Switch {_name} {_count} inputs", "Switch", input_selection="int",
              **{f"value_{i + 1}": _t for i in range(_count)})
        def _(sel, *values):
            def pick(sel, values):
                if len(values) == 2:
                    return _if(sel > 0, values[1], values[0])
                half = len(values) // 2
                return _if(sel < half, pick(sel, values[:half]), pick(sel - half, values[half:]))
            return pick(sel, values)
_B = None
