"""Shared pieces of the Paint filters and generators.

Conventions (ours, decision 6). Port names say what the Paint workspace gives them (paint/graph_nodes.py), in a
graph they are connected like any image:
- `input` -> `output`: a filter of one channel (Paint: the stack below the effect, of the channel being
  computed); the stack's coverage (alpha) is kept around it.
- ports named after the material channels (`basecolor`, `normal`, `roughness`, `metallic`, `height`,
  `ambient_occlusion`, `opacity`, `emissive`... the material convention, nodes/materials/_base.py): a
  filter of the whole material (MatFX, MatFinish...): each input is the stack below of that channel, each
  channel takes the output of its name (channels without one are left as they are).
- mesh maps: the ports of `MESH_MAPS` (`mesh_inputs`), the Texture Set's baked or computed maps; unconnected
  they read a neutral value.
- any other image input: a custom texture (Paint: a resource slot of the effect).
- generators write grayscale `output` unless they make color (UV Random Color...) or a material.
- the "Seed" parameter is the node's Random Seed (base parameter, `p.seed_`).

Each node's `.wgsl` defines `fn effect(c: vec2i) -> vec4f`; the `main` that stores it is added here.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import blur2d, gaussian, source

DIR = Path(__file__).parent
LIB = (DIR / "lib.wgsl").read_text(encoding="utf-8")
DOC = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/"
MAIN = """
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    store(c, effect(c));
}"""
SEED = {"Seed": "the node's Random Seed (base parameter)"}

FILTERS = "Paint/Filters"
GENERATORS = "Paint/Generators"


def F(label, default, lo=None, hi=None):
    return Param(ParamKind.FLOAT, float(default), label, lo, hi)


def B(label, default=False):
    return Param(ParamKind.BOOL, bool(default), label)


def E(label, choices, default=None):
    return Param(ParamKind.ENUM, default or choices[0], label, choices=tuple(choices))


def C(label, default):
    return Param(ParamKind.FLOAT3, [float(x) for x in default], label, 0.0, 1.0, editor="Color")


def I(label, default, lo=None, hi=None):  # noqa: E743 - the INT shorthand next to F, B, E, C
    return Param(ParamKind.INT, int(default), label, lo, hi)


# Mesh map inputs: port -> (Texture Set mesh map slot, port type, label, value read when unconnected). The ports
# of the maps whose slot name is also a material channel are prefixed `baked_`.
# Computed (not baked) from the mesh: `uv_islands` (RG = two random numbers per UV island, B = 0,
# A = 1 inside the islands and 0 outside the UVs) and `texel_density` (log2 of the texel density relative to the
# Texture Set's mean: 0 = mean, 1 = twice as many texels per surface unit).
MESH_MAPS = {
    "curvature": ("curvature", PortType.GRAY, "Curvature", (0.5, 0.5, 0.5, 1.0)),
    "baked_ambient_occlusion": ("ambient_occlusion", PortType.GRAY, "Ambient Occlusion", (1.0, 1.0, 1.0, 1.0)),
    "thickness": ("thickness", PortType.GRAY, "Thickness", (0.5, 0.5, 0.5, 1.0)),
    "baked_height": ("height", PortType.GRAY, "Height", (0.5, 0.5, 0.5, 1.0)),
    "id": ("id", PortType.COLOR, "ID", (0.0, 0.0, 0.0, 1.0)),
    "world_space_normals": ("world_space_normals", PortType.COLOR, "World Space Normals", (0.5, 1.0, 0.5, 1.0)),
    "baked_normal": ("normal", PortType.COLOR, "Normal", (0.5, 0.5, 1.0, 1.0)),
    "position": ("position", PortType.COLOR, "Position", (0.5, 0.5, 0.5, 1.0)),
    "uv_islands": ("uv_islands", PortType.COLOR, "UV Islands", (0.0, 0.0, 0.0, 1.0)),
    "texel_density": ("texel_density", PortType.GRAY, "Texel Density", (0.0, 0.0, 0.0, 1.0)),
}


def mesh_inputs(*ports) -> dict[str, Input]:
    return {s: Input(MESH_MAPS[s][1], MESH_MAPS[s][2], MESH_MAPS[s][3]) for s in ports}


class PaintNode(Node):
    """A node of this area: its shader is `lib.wgsl` + its `includes` + its own `.wgsl` + `main`."""

    includes: tuple = ()
    doc: str = ""  # the documentation page

    def wgsl(self) -> str:
        own = Path(inspect.getfile(type(self))).with_suffix(".wgsl").read_text(encoding="utf-8")
        return "\n".join([LIB, *((DIR / f"{i}.wgsl").read_text(encoding="utf-8") for i in self.includes), own, MAIN])


class Filter(PaintNode):
    category = FILTERS + "/Standard"
    inputs = {"input": Input(label="Input")}
    outputs = {"output": Output()}


class Generator(PaintNode):
    category = GENERATORS
    outputs = {"output": Output(PortType.GRAY)}


def helper(type_: str, shader: str, inputs: dict, outputs: dict, params: dict | None = None,
           internal: dict | None = None, main: bool = False) -> Node:
    """A pass with its own shader file of this folder (with `main` added when it defines `effect`), not in the
    library."""

    def wgsl(self):
        return "\n".join([LIB, (DIR / shader).read_text(encoding="utf-8")] + ([MAIN] if main else []))

    cls = type(f"_{type_}", (Node,), {"type": type_, "registered": False, "inputs": inputs, "outputs": outputs,
                                      "params": params or {}, "internal_uniforms": internal or {}, "wgsl": wgsl})
    return cls()


def soft_blur(ctx, job, tex, gray: bool, radius: float, wrap: bool = True):
    """Gaussian of a texture, sigma `radius` in its pixels at 256 (scaled with its size); `wrap` samples across the
    borders. The result has the texture's size."""
    seen = job if wrap else dataclasses.replace(job, tiling=0)
    return gaussian(ctx, seen, tex, gray, float(radius) * tex.width / 256.0)


def blur(ctx, job, intensity: float, wrap: bool = True, dst=None):
    """Gaussian blur of the job's input, sigma `intensity` (texels at 256 px, scaled with the size), into `dst`
    (default: the job's output); `wrap` off clamps at the borders. Returns the texture written."""
    gray = job.is_gray("output")
    seen = job if wrap else dataclasses.replace(job, tiling=0)
    return blur2d(ctx, seen, source(ctx, job, "input"), gray, float(intensity) * job.size[0] / 256.0,
                  dst=dst if dst is not None else ctx.ev.texture(job, "output"))
