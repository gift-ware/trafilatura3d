"""FX-Map graphs: a quad-tree of Quadrant nodes (each subdivides its cell into four
and can draw a pattern), Iterate nodes (repeat a branch, $number = repetition) and Switch nodes (pick
one of two branches). Walking the tree on the CPU yields the list of patterns to draw, in drawing
order (top to bottom, then left to right); the FX-Map node draws them on the GPU.

An FX-Map graph is {"nodes": [{"id", "type", "params", "functions": {param: function graph},
"children": {slot: node id}, "position"}], "root": node id}. Every parameter may be a function of
$pos, $depth, $depthpow2, $number, $size, $sizelog2, $time and the graph's parameters.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from trafilatura3d.functions import BUILTINS, PARAM_TYPES, FunctionError, FunctionGraph, evaluate, set_variables, to_param
from trafilatura3d.nodes.base import Param, ParamKind

# The .sbs patterntype is the index here (the library's drop-down lists of patterns: Tile Sampler, Splatter Circular).
PATTERNS = ("No Pattern", "Input image", "Square", "Disc", "Paraboloid", "Bell", "Gaussian", "Thorn", "Pyramid",
            "Brick", "Gradation", "Waves", "Half Bell", "Ridged Bell", "Crescent", "Capsule", "Cone",
            "Gradation w. offset", "Hemisphere")
# The Quadrant's blending, in .sbs order (values 0, 1, 2): "Add Sub" (the default) is a
# signed addition, pattern x Color / Luminosity added (a negative colour subtracts; Tile Random draws v - background
# in it to get lerp(background, v, pattern)), Max keeps the brighter, Alpha blend lays the colour over what is below
# with the pattern (x the colour's alpha) as opacity.
BLENDING = ("Add Sub", "Max", "Alpha blend")
MAX_DEPTH = 12
MAX_DRAWS = 1 << 20

FX_PARAMS = {
    "quadrant": {
        "color_luminosity": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Color / Luminosity"),
        "branch_offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Branch Offset", -1.0, 1.0),
        "pattern": Param(ParamKind.ENUM, "No Pattern", "Pattern", choices=PATTERNS),
        "pattern_offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Pattern Offset", -1.0, 1.0),
        "pattern_size": Param(ParamKind.FLOAT2, [1.0, 1.0], "Pattern Size", 0.0, 4.0),
        "pattern_rotation": Param(ParamKind.FLOAT, 0.0, "Pattern Rotation", 0.0, 1.0),  # turns
        "pattern_variation": Param(ParamKind.FLOAT, 0.0, "Pattern Variation", 0.0, 1.0),
        "blending_mode": Param(ParamKind.ENUM, "Add Sub", "Blending Mode", choices=BLENDING),
        "random_seed": Param(ParamKind.INT, 0, "Random Seed", 0, 10000),
        "inherit_random": Param(ParamKind.BOOL, True, "Inherit Random"),
        "input_image": Param(ParamKind.INT, 0, "Input image", 0, 7),  # which input "Input image" draws
        "input_image_alpha": Param(ParamKind.ENUM, "Use alpha", "Input image alpha",
                                   choices=("Use alpha", "Ignore alpha")),
        "input_image_filtering": Param(ParamKind.ENUM, "Bilinear", "Input image filtering",
                                       choices=("Bilinear", "Nearest")),
    },
    "iterate": {
        # Number Added defaults to 1: an Iterate that leaves it unset merges two branches (no $number, Add Sub);
        # larger counts (2, 5, 16...) are set explicitly.
        "iterations": Param(ParamKind.INT, 1, "Iterations", 0, 1000),
        "random_seed": Param(ParamKind.INT, 0, "Random Seed", 0, 10000),
    },
    "switch": {
        "selector": Param(ParamKind.INT, 0, "Selector", 0, 1),  # 0: Input 1, otherwise Input 2
    },
}
SLOTS = {"quadrant": ("q1", "q2", "q3", "q4"), "iterate": ("pass_through", "repeated"),
         "switch": ("input_1", "input_2")}
SLOT_LABELS = {"q1": "1", "q2": "2", "q3": "3", "q4": "4", "pass_through": "Pass-through", "repeated": "Repeated",
               "input_1": "Input 1", "input_2": "Input 2"}
FX_VARIABLES = {"$pos": "float2", "$depth": "float", "$depthpow2": "float", "$number": "float",
                "$size": "float2", "$sizelog2": "float2", "$time": "float"}

DEFAULT_FXMAP = {"nodes": [{"id": "quadrant1", "type": "quadrant", "params": {"pattern": "Disc"},
                            "children": {}, "position": [0, 0]}], "root": "quadrant1"}


@dataclass
class Draw:
    pattern: int
    center: tuple[float, float]
    size: tuple[float, float]
    rotation: float  # turns
    variation: float
    color: tuple[float, float, float, float]
    blend: int
    image: int
    filtering: int  # 0 bilinear, 1 nearest
    alpha: int  # 0 use alpha, 1 ignore
    depth: int


class FxMapError(ValueError):
    pass


def _hash(*parts) -> int:
    return int.from_bytes(hashlib.sha1(":".join(map(str, parts)).encode()).digest()[:4], "little")


def walk(data: dict, graph_vars: dict, graph_types: dict, size: tuple[int, int], seed: int,
         node_vars: dict | None = None, system: dict | None = None) -> list[Draw]:
    """The patterns of an FX-Map graph, in drawing order. `graph_vars`/`graph_types`: the graph's
    parameters as function values and types; `node_vars`: what the FX-Map node's own functions Set; `system`: the
    node's system variables ($tiling...; $size and $sizelog2 are the walk's). The functions
    share one scope through the walk: what one Sets is seen by those evaluated after it (an Iterate's
    child, the next iteration)."""
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    root = data.get("root")
    if root is None or root not in nodes:
        return []
    variables_types = {**BUILTINS, **FX_VARIABLES, **graph_types}
    compiled: dict[tuple, FunctionGraph] = {}
    draws: list[Draw] = []
    node_vars = node_vars or {}
    outer = frozenset(node_vars) | set_variables(f for n in nodes.values() for f in n.get("functions", {}).values())
    env = {**(system or {}), "$size": [float(size[0]), float(size[1])],
           "$sizelog2": [math.log2(size[0]), math.log2(size[1])], "$time": 0.0, **graph_vars, **node_vars}

    def params(node, env, node_seed):
        """The node's parameters, functions evaluated top to bottom in one shared environment."""
        kind = node["type"]
        values = {}
        for name, p in FX_PARAMS[kind].items():
            value = node.get("params", {}).get(name, p.default)
            f = node.get("functions", {}).get(name)
            if f is not None and p.kind in PARAM_TYPES:
                key = (node["id"], name)
                if key not in compiled:
                    try:
                        compiled[key] = FunctionGraph(f, variables_types, outer)
                    except FunctionError as e:
                        raise FxMapError(f"{node['id']}.{name}: {e}") from None
                value = to_param(evaluate(compiled[key], env, seed=node_seed, shared=True), p)
            values[name] = value
        return values

    def visit(node_id, depth, origin, cell, offset, number, inherited_seed, path):
        if node_id not in nodes or depth > MAX_DEPTH or len(draws) >= MAX_DRAWS:
            return
        node = nodes[node_id]
        kind = node["type"]
        children = node.get("children", {})
        center = (origin[0] + cell / 2, origin[1] + cell / 2)
        env.update({"$pos": list(center), "$depth": float(depth), "$depthpow2": 1.0 / (1 << depth),
                    "$number": float(number)})
        if kind == "quadrant":
            own_seed = node.get("params", {}).get("random_seed", 0)
            node_seed = inherited_seed if node.get("params", {}).get("inherit_random", True) else own_seed + seed
            v = params(node, env, node_seed)
            off = (offset[0] + v["branch_offset"][0], offset[1] + v["branch_offset"][1])
            pattern = PATTERNS.index(v["pattern"])
            if pattern:
                draws.append(Draw(pattern, (center[0] + off[0] + v["pattern_offset"][0],
                                            center[1] + off[1] + v["pattern_offset"][1]),
                                  (v["pattern_size"][0] * cell, v["pattern_size"][1] * cell), v["pattern_rotation"],
                                  v["pattern_variation"], tuple(v["color_luminosity"]), BLENDING.index(v["blending_mode"]),
                                  v["input_image"], int(v["input_image_filtering"] == "Nearest"),
                                  int(v["input_image_alpha"] == "Ignore alpha"), depth))
            half = cell / 2
            for k, slot in enumerate(SLOTS["quadrant"]):
                if slot in children:
                    visit(children[slot], depth + 1, (origin[0] + (k % 2) * half, origin[1] + (k // 2) * half),
                          half, off, number, _hash(node_seed, k), path + (k,))
        elif kind == "iterate":
            v = params(node, env, inherited_seed)
            if "pass_through" in children:
                visit(children["pass_through"], depth, origin, cell, offset, number, inherited_seed, path + ("p",))
            if "repeated" in children:
                for i in range(max(0, int(v["iterations"]))):
                    visit(children["repeated"], depth, origin, cell, offset, i,
                          _hash(inherited_seed, v["random_seed"], i), path + (i,))
        elif kind == "switch":
            v = params(node, env, inherited_seed)
            slot = "input_1" if int(v["selector"]) == 0 else "input_2"
            if slot in children:
                visit(children[slot], depth, origin, cell, offset, number, inherited_seed, path + (slot,))

    visit(root, 0, (0.0, 0.0), 1.0, (0.0, 0.0), 0, seed, ())
    return draws


def levels(data: dict) -> int:
    """The quad-tree levels of an FX-Map graph: the deepest Quadrant any branch can reach, plus one (Switch
    selectors and Iterate counts aside)."""
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    memo: dict[str, int] = {}

    def below(node_id, stack):  # the levels from this node down
        if node_id not in nodes or node_id in stack:
            return 0
        if node_id not in memo:
            node = nodes[node_id]
            deepest = max((below(c, stack | {node_id}) for c in node.get("children", {}).values()), default=0)
            memo[node_id] = min(MAX_DEPTH + 1, deepest + (node["type"] == "quadrant"))
        return memo[node_id]
    return max(1, below(data.get("root"), frozenset()))


def level_weights(roughness: float, depth, levels: int) -> np.ndarray:
    """The FX-Map node's Roughness (`hurst` in .sbs files) as the opacity of patterns at quad-tree `depth` (array) in
    a tree of `levels` levels: r^depth normalised so the levels add up to 1, i.e. 1/levels each at 1 (the default);
    0 turns it off (1 everywhere). A Global opacity equal to the level count (8 for 8 levels) gives each level
    full weight at 1."""
    d = np.asarray(depth, np.float64)
    r = float(roughness)
    if r == 0.0:
        return np.ones_like(d)
    if r == 1.0:
        return np.full_like(d, 1.0 / levels)
    return (1.0 - r) * r ** d / (1.0 - r ** levels)


# `level_weights` for shaders: the weight of `level` (from 0) of `levels` at roughness `r`.
ROUGHNESS_WGSL = """
fn roughness_weight(r: f32, level: f32, levels: f32) -> f32 {
    if (r == 0.0) { return 1.0; }
    if (r == 1.0) { return 1.0 / levels; }
    return (1.0 - r) * pow(r, level) / (1.0 - pow(r, levels));
}
"""


def check(data: dict) -> list[str]:
    """Structural problems of an FX-Map graph (types, children, cycles)."""
    problems = []
    nodes = {n.get("id"): n for n in data.get("nodes", [])}
    for k, n in nodes.items():
        if n.get("type") not in FX_PARAMS:
            problems.append(f"{k}: unknown FX-Map node {n.get('type')!r}")
            continue
        for slot, child in n.get("children", {}).items():
            if slot not in SLOTS[n["type"]] or child not in nodes:
                problems.append(f"{k}: bad child {slot!r} -> {child!r}")
    return problems


def graph_variables(graph_params: dict) -> tuple[dict, dict]:
    """(values, types) of the graph parameters for FX-Map functions, from a job's graph_params."""
    values, types = {}, {}
    for pid, (kind_name, value) in graph_params.items():
        kind = ParamKind(kind_name)
        if kind in PARAM_TYPES:
            values[pid], types[pid] = value, PARAM_TYPES[kind]
    return values, types
