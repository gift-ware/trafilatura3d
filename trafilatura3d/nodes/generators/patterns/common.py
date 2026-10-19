"""Shared base, parameter helpers and WGSL generators of the pattern nodes.

Every shader of this folder is `nodes/shapes.wgsl` (the built-in pattern shapes, shared with FX-Map) +
`lib.wgsl` (helpers) + the files in
`libs` (e.g. `tiles.wgsl`, the tile-grid engine) + generated pattern-input samplers + the node's
own `<module>.wgsl` (or `shader`)."""

import math
from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import downsample

HERE = Path(__file__).parent
SHAPES_WGSL = (HERE.parent.parent / "shapes.wgsl").read_text(encoding="utf-8")  # the built-in shapes (FX-Map's too)
CATEGORY = "Texture generators/Patterns"

F, I, B, E, C = ParamKind.FLOAT, ParamKind.INT, ParamKind.BOOL, ParamKind.ENUM, ParamKind.COLOR
F2, F3, F4 = ParamKind.FLOAT2, ParamKind.FLOAT3, ParamKind.FLOAT4

GRAY_OUT = {"output": Output(PortType.GRAY, "Output")}

# Built-in pattern shapes, in this order; the shader index is the position in this tuple
# (nodes/shapes.wgsl `shape()`: the FX-Map pattern index - 2).
SHAPES = ("Square", "Disc", "Paraboloid", "Bell", "Gaussian", "Thorn", "Pyramid", "Brick", "Gradation",
          "Waves", "Half Bell", "Ridged Bell", "Crescent", "Capsule", "Cone", "Gradation w. offset", "Hemisphere")
FILTERING = ("Bilinear + Mipmaps", "Bilinear", "Nearest")
SYMMETRY = ("Horizontal + Vertical", "Horizontal", "Vertical")


def pattern_choices(first: str, **rename) -> tuple[str, ...]:
    """The Pattern enum of the tile nodes: the image-input choice first (shader: index 0), then the
    shapes (shader: index - 1), with the node's own spelling of some names."""
    return (first,) + tuple(rename.get(s.replace(" ", "_").lower(), s) for s in SHAPES)


def nse(label: str = "Non Square Expansion") -> dict:
    return {"non_square_expansion": Param(B, True, label)}


def color_variant() -> dict:
    """The Color variant of a pattern that has both (Tile Sampler Color, Splatter Color...): a colour output
    also when every input is grayscale. Off, the output is colour only when a colour input is connected."""
    return {"color_variant": Param(B, False, "Color Variant")}


def tiling(default: int = 1) -> dict:
    return {"tiling": Param(I, default, "Tiling", 1, 16)}


def prefiltered(ctx, job, ports, footprint: float) -> dict:
    """The pattern inputs `ports` for Bilinear + Mipmaps as copies reduced by the largest power of 2 that an output
    pixel still covers on every tile (port -> (texture, is_gray), for `srcs`; inputs that need no reduction are left
    out). `footprint`: an output pixel in a pattern's uv on the largest tile the node can draw. pat_sample then reads
    a few nearby texels of a small copy instead of up to 16 scattered over the whole input."""
    out = {}
    for port in ports:
        s = job.inputs.get(port)
        if not s:
            continue
        tex, gray = ctx.ev.texture(*s), s[0].is_gray(s[1])
        d = 1 << max(math.floor(math.log2(max(footprint * max(tex.width, tex.height), 1.0))), 0)
        if d > 1:
            out[port] = (downsample(ctx, job, tex, gray, (d, d)), gray)
    return out


def pattern_sampler(ports: list[str]) -> str:
    """WGSL sampling the pattern inputs `ports` by index: `pat_sample(k, uv, filtering, fw)` reads
    input k at uv in [0, 1] (0 outside), Filtering index as in FILTERING, `fw` = the output pixel
    footprint in uv of the input (for the mipmapped mode)."""
    cases, sizes = [], []
    for i, port in enumerate(ports):
        cases.append(f"case {i}: {{ let s = vec2f(size_{port}()); if (nearest) {{ return texel_{port}("
                     f"vec2i(floor(uv * s))); }} return sample_{port}(clamp(uv, 0.5 / s, 1.0 - 0.5 / s)); }}")
        sizes.append(f"case {i}: {{ return vec2f(size_{port}()); }}")
    return f"""
fn pat_texel(k: i32, uv: vec2f, nearest: bool) -> vec4f {{
    switch k {{ {' '.join(cases)} default: {{ return vec4f(0.0); }} }}
}}
fn pat_size(k: i32) -> vec2f {{
    switch k {{ {' '.join(sizes)} default: {{ return vec2f(1.0); }} }}
}}
fn pat_sample(k: i32, uv: vec2f, filtering: i32, fw: f32) -> vec4f {{
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {{ return vec4f(0.0); }}
    if (filtering == 2) {{ return pat_texel(k, uv, true); }}
    let s = pat_size(k);
    let n = select(1, clamp(i32(ceil(fw * max(s.x, s.y))), 1, 4), filtering == 0);
    if (n == 1) {{ return pat_texel(k, uv, false); }}
    var acc = vec4f(0.0);
    for (var j = 0; j < n; j++) {{
        for (var i = 0; i < n; i++) {{
            let o = (vec2f(f32(i), f32(j)) + 0.5) / f32(n) - 0.5;
            acc += pat_texel(k, clamp(uv + o * fw, vec2f(0.0), vec2f(1.0)), false);
        }}
    }}
    return acc / f32(n * n);
}}
"""


class PatternNode(Node):
    """A pattern node: category, shared WGSL libraries, optional pattern-input samplers."""

    category = CATEGORY
    outputs = GRAY_OUT
    libs: tuple[str, ...] = ()  # extra shared .wgsl files of this folder, after lib.wgsl
    pattern_ports: tuple[str, ...] = ()  # inputs sampled by index through pat_sample()
    shader = ""  # .wgsl file of this folder (default: the module's)
    seamless = True

    def output_types(self, input_types, values):
        types = super().output_types(input_types, values)
        if values.get("color_variant"):
            return {name: PortType.COLOR if self.outputs[name].type is PortType.ANY else t for name, t in types.items()}
        return types

    def wgsl(self) -> str:
        parts = [SHAPES_WGSL, (HERE / "lib.wgsl").read_text(encoding="utf-8")] + [(HERE / f).read_text(encoding="utf-8") for f in self.libs]
        if self.pattern_ports:
            parts.append(pattern_sampler(list(self.pattern_ports)))
        parts.append((HERE / self.shader).read_text(encoding="utf-8") if self.shader else super().wgsl())
        return "\n".join(parts)


__all__ = ["B", "C", "CATEGORY", "E", "F", "F2", "F3", "F4", "FILTERING", "GRAY_OUT", "HERE", "I", "Input",
           "Output", "Param", "ParamKind", "PatternNode", "PortType", "SHAPES", "SYMMETRY", "color_variant", "nse",
           "pattern_choices", "pattern_sampler", "tiling"]
