"""The library's nodes where the Paint workspace has the same kind of resource (decision 23): the texture
generators (Noises, Patterns) are the Paint procedurals (a fill channel's source); the Paint filters and generators
(`nodes/paint/`) then the Graph Filters are filters (their main input takes the stack below the effect); the Paint
generators then the Graph mesh based Mask Generators are generators (their inputs named after a mesh map are given
the Texture Set's baked map). A Graph node with the label of a Paint one is left out: the Paint one is used.

What an effect gives each input port (`role`, conventions of nodes/paint/_lib.py): `input` (or a Graph
filter's main input) the stack below of the channel being computed; a port named after a material channel the
stack below of that channel at the same point (a filter of the whole material); a mesh map port the Texture
Set's map; any other image input is a resource slot of the effect (a texture or a procedural, `Effect.inputs`).
The channel takes the output named after it, else `output`; a material filter without an output for the
channel leaves it as it is.

Nodes that need what an effect cannot give (strings, function graphs, FX-Maps, value ports) are left out.
"""

from __future__ import annotations

from trafilatura3d.evaluator import Job
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.base import ParamKind, PortType
from trafilatura3d.nodes.paint._lib import FILTERS as PAINT_FILTERS
from trafilatura3d.nodes.paint._lib import GENERATORS as PAINT_GENERATORS
from trafilatura3d.nodes.paint._lib import MESH_MAPS

PREFIX = "node:"  # an effect's or a channel source's `procedural` naming a library node
SEED = "$seed"  # in the values: the node's Random Seed (base parameter), 0 by default
_UNUSABLE = {ParamKind.STRING, ParamKind.FUNCTION, ParamKind.FXMAP}

# Input port names of the mask generators -> Texture Set mesh map slot.
MESH_PORTS = {
    "curvature": "curvature", "baked_curvature": "curvature", "ambient_occlusion": "ambient_occlusion",
    "baked_ambient_occlusion": "ambient_occlusion", "world_space_normal": "world_space_normals",
    "normal_world_space": "world_space_normals", "baked_normal_ws": "world_space_normals",
    "baked_world_space_normal": "world_space_normals", "baked_normal_world_space": "world_space_normals",
    "position": "position", "baked_position": "position", "position_map": "position", "thickness": "thickness",
    "baked_height": "height", "color_id": "id",
    **{port: m[0] for port, m in MESH_MAPS.items()},  # the Paint filters and generators (nodes/paint/_lib.py)
}

# Material channel ports (the material convention, nodes/materials/_base.py) -> Paint channel id.
CHANNEL_PORTS = {
    "basecolor": "basecolor", "diffuse": "diffuse", "normal": "normal", "specular": "specular",
    "emissive": "emissive", "glossiness": "glossiness", "roughness": "roughness", "metallic": "metallic",
    "specular_level": "specularlevel", "ambient_occlusion": "ambientocclusion", "height": "height",
    "opacity": "opacity",
}
PORT_OF_CHANNEL = {c: p for p, c in CHANNEL_PORTS.items()}


def _usable(node) -> bool:
    if any(p.kind in _UNUSABLE for p in node.params.values()):
        return False
    if any(i.type is PortType.VALUE for i in node.inputs.values()):
        return False
    images = [o for o, spec in node.outputs.items() if spec.type is not PortType.VALUE]  # value outputs unused
    return "output" in images or any(o in CHANNEL_PORTS for o in images)


def role(node, port: str, kind: str) -> tuple[str, str | None]:
    """What an effect of `kind` ("filter" | "generator") gives the node's input `port`: ("stack", None) the
    stack below of the channel computed, ("channel", channel id) the stack below of another channel, ("mesh",
    slot) a mesh map, ("resource", None) a resource slot of the effect."""
    if kind == "filter":
        if port == "input" or (port == node.primary_input() and not any(p in CHANNEL_PORTS for p in node.inputs)):
            return "stack", None
        if port in CHANNEL_PORTS:
            return "channel", CHANNEL_PORTS[port]
    if port in MESH_PORTS:
        return "mesh", MESH_PORTS[port]
    if port in CHANNEL_PORTS:
        return "channel", CHANNEL_PORTS[port]
    return "resource", None


def resource_ports(node, kind: str) -> list[str]:
    return [p for p in node.inputs if role(node, p, kind)[0] == "resource"]


def output_for(node, channel: str) -> str | None:
    """The output a channel takes (None: the node leaves the channel as it is)."""
    port = PORT_OF_CHANNEL.get(channel)
    if port in node.outputs:
        return port
    return "output" if "output" in node.outputs else None


def procedurals() -> list:
    """Texture generators usable as a channel's source: no input to give."""
    return sorted((n for n in NODES.values() if n.category.startswith("Texture generators") and _usable(n)
                   and not n.inputs), key=lambda n: (n.category, n.label))


def _paint_first(paint: str, graph: str, filters: bool) -> list:
    def ok(n):
        return _usable(n) and (not filters or (n.inputs and n.primary_input() is not None))
    own = sorted((n for n in NODES.values() if n.category.startswith(paint) and ok(n)),
                 key=lambda n: (n.category, n.label))
    labels = {n.label for n in own}
    return own + sorted((n for n in NODES.values() if n.category.startswith(graph) and ok(n)
                         and n.label not in labels), key=lambda n: (n.category, n.label))


def filters() -> list:
    return _paint_first(PAINT_FILTERS, "Filters/", True)


def generators() -> list:
    return _paint_first(PAINT_GENERATORS, "Mesh based generators/Mask Generators", False)


def group_of(node) -> str:
    """The node's group in the menus: the Paint ones (Standard, Advanced… / Generators) or the Graph category."""
    return node.category.split("/")[-1]


def node_of(procedural: str):
    """The node of a "node:<type>" reference, or None."""
    return NODES.get(procedural[len(PREFIX):]) if procedural.startswith(PREFIX) else None


def values_of(node, values: dict) -> dict:
    out = {}
    for name, p in node.params.items():
        v = values.get(name, p.default)
        try:
            v = p.validate(v) if v is not None else p.default
        except ValueError:
            v = p.default
        out[name] = v
    return out


def job(compiler, node, values: dict, inputs: dict, size, dep: str | dict, key_extra=None) -> Job:
    """A job of a library node for the stack: `inputs` {port: (job, output)}; `dep` the depth of every output,
    or {output: depth}."""
    from trafilatura3d.paint.stack import _key
    full = values_of(node, values)
    node = node.instantiate(full)
    ins = {port: inputs.get(port) for port in node.inputs}
    types = {port: (src[0].output_types[src[1]] if src else None) for port, src in ins.items()}
    out_types = node.output_types(types, full)
    try:
        extra = node.extra(full, None)
    except Exception:  # noqa: BLE001 - nodes reading package resources have nothing to read here
        extra = {}
    seed = float(values.get(SEED, 0))
    depths = dep if isinstance(dep, dict) else {o: dep for o in node.outputs}
    key = _key(compiler.udim, node.type, full, {p: (s[0].key if s else None) for p, s in ins.items()}, list(size),
               depths, key_extra, extra, seed)
    return Job(key=key, path=node.type, node=node, values=full, inputs=ins, output_types=out_types,
               size=tuple(size), depths=depths, tiling=3, seed=seed, extra=extra, keep=False,
               partition=compiler.partition(size))
