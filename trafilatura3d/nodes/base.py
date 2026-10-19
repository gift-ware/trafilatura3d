"""Node contract: typed inputs and outputs, typed parameters, one WGSL compute shader per node.

A node is a stateless definition (one instance per type, in the registry). Its shader is the
`.wgsl` file next to its module; the header is generated from the declarations for every variant
(which inputs are connected and whether they are grayscale or color, output formats):

- `p` — the parameters as a struct (`p.<name>`), plus built-ins: `p.size_` (output size in pixels),
  `p.seed_` (random seed), `p.tiling_` (bit 0: horizontal, bit 1: vertical), `p.origin_` (`origin()`: the corner
  of the region a `pointwise` node is dispatched over, Evaluator.recook_region; 0 otherwise);
- for each input port `x`: `load_x(c)` (texel at output pixel `c`; bilinear at its centre when sizes differ),
  `sample_x(uv)` (bilinear at uv in [0,1], wrapping or clamping as the tiling says), `size_x()`,
  constants `X_CONNECTED` and `X_GRAY`; grayscale inputs read as (v, v, v, 1);
- for each output `y`: `store_y(c, v)` (quantized and clamped to the output format), and
  `store(c, v)` for the first output; `size()` is the output size;
- `lut_<name>(t)` for gradient and curve parameters (baked to a 256-texel lookup table).
"""

from __future__ import annotations

import re

import inspect
import json
import struct
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from trafilatura3d.evaluator import CookContext, Job

# Our node types whose Output Format is fixed whatever they inherit, as in the library graph each follows (the one
# the library shows): type -> format. A node's own `base_defaults` (a format its data needs) wins.
DEPTHS: dict[str, str] = json.loads(Path(__file__).with_name("depths.json").read_text(encoding="utf-8"))


class PortType(Enum):
    GRAY = "grayscale"
    COLOR = "color"
    ANY = "any"  # follows what is connected: color if any `any` input is color
    VALUE = "value"  # a single value (Value processor, Input value…): a 1x1 rgba32float texture
    # A grayscale value and its coverage (the Paint stack, grayscale channels): two channels stored, read as
    # (v, v, v, a). Only a job's output type, no port has it.
    GRAY_ALPHA = "grayscale_alpha"

    def accepts(self, got: PortType) -> bool:
        """Whether an input of this type takes a connection of type `got`."""
        if self is PortType.ANY:
            return got in (PortType.GRAY, PortType.COLOR)
        return self is got


@dataclass(frozen=True)
class Input:
    type: PortType = PortType.ANY
    label: str = ""
    default: tuple = (0.0, 0.0, 0.0, 0.0)  # read when nothing is connected


@dataclass(frozen=True)
class Output:
    type: PortType = PortType.ANY
    label: str = ""


class ParamKind(Enum):
    FLOAT = "float"
    FLOAT2 = "float2"
    FLOAT3 = "float3"
    FLOAT4 = "float4"
    INT = "int"
    INT2 = "int2"
    BOOL = "bool"
    ENUM = "enum"
    COLOR = "color"
    STRING = "string"  # not passed to the shader
    GRADIENT = "gradient"  # [{"pos": t, "color": [r, g, b, a]}...] -> lut_<name>(t)
    CURVE = "curve"  # [[x, y] or [x, y, lx, ly, rx, ry] (Bezier handles)...] in [0, 1] -> lut_<name>(t).r
    FUNCTION = "function"  # a function graph (trafilatura3d/functions.py), compiled by the node itself
    FXMAP = "fxmap"  # an FX-Map graph (trafilatura3d/fxmap.py)


# Uniform layout per kind: WGSL type, struct format, size in floats, alignment in bytes.
_UNIFORM = {
    ParamKind.FLOAT: ("f32", "f", 4),
    ParamKind.FLOAT2: ("vec2f", "2f", 8),
    ParamKind.FLOAT3: ("vec3f", "3f", 16),
    ParamKind.FLOAT4: ("vec4f", "4f", 16),
    ParamKind.COLOR: ("vec4f", "4f", 16),
    ParamKind.INT: ("i32", "i", 4),
    ParamKind.INT2: ("vec2i", "2i", 8),
    ParamKind.BOOL: ("i32", "i", 4),
    ParamKind.ENUM: ("i32", "i", 4),
}
# The gradient editor: Linear, Flat Tangents (Midpoint), Smooth (Smoothness), No Interpolation.
GRADIENT_INTERPOLATIONS = ("linear", "flat", "smooth", "none")
_VECTOR_LEN = {ParamKind.FLOAT2: 2, ParamKind.FLOAT3: 3, ParamKind.FLOAT4: 4, ParamKind.INT2: 2}
# The editors (UI controls) a graph input of each kind can have; the first is the default.
EDITORS = {
    ParamKind.FLOAT: ("Slider", "Angle"),
    ParamKind.INT: ("Slider", "Drop down list", "Random seed", "Size pow 2"),
    ParamKind.BOOL: ("Togglebutton",),
    ParamKind.FLOAT2: ("Slider", "Position"),
    ParamKind.FLOAT3: ("Slider", "Color"),
    ParamKind.FLOAT4: ("Slider", "Color"),
    ParamKind.INT2: ("Slider", "Size pow 2"),
    ParamKind.COLOR: ("Color",),
    ParamKind.STRING: ("Text",),
    ParamKind.ENUM: ("Drop down list",),
}
LUT_SIZE = 256
# The side of a compute workgroup (the shaders' `@workgroup_size(WG, WG)`, the dispatches' groups): 16 x 16 makes a
# texture pass at 4096 0.33 ms where 8 x 8 made it 0.5-0.9 (any code beyond the read and the store doubled it).
WORKGROUP = 16

BUILTINS =[("size_", ParamKind.FLOAT2), ("seed_", ParamKind.FLOAT), ("tiling_", ParamKind.INT),
            ("origin_", ParamKind.FLOAT2), ("clip_", ParamKind.FLOAT4)] + [
    (f"pages{i}_", ParamKind.INT) for i in range(8)]  # the pages computed (Variant.pages): a bit per 1/16 x 1/16
PAGE_GRID = 16  # Variant.pages: the output cut in PAGE_GRID x PAGE_GRID pages, bit y * 16 + x of pages0_..pages7_
MAIN = "fn main(@builtin(global_invocation_id) id: vec3u) {"
# Plus one "q_<output>" per output: quantization levels - 1 (255, 65535) or 0 for float formats.

INT_MATH_WGSL = (Path(__file__).parent / "int_math.wgsl").read_text(encoding="utf-8")  # also for the paint passes
COMMON_WGSL = INT_MATH_WGSL + (Path(__file__).parent / "common.wgsl").read_text(encoding="utf-8")


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class Param:
    kind: ParamKind
    default: Any
    label: str = ""
    min: float | None = None  # slider range (values outside are allowed)
    max: float | None = None
    choices: tuple[str, ...] = ()  # ENUM: stored by name, passed to the shader as its index
    group: str = ""  # "/" separates sub groups (collapsed together in the Properties)
    # The properties of an exposed graph input (nodes leave them empty):
    description: str = ""  # tooltip
    editor: str = ""  # UI editor, see EDITORS ("" = the kind's default)
    step: float | None = None
    clamp: bool = False  # Min / Max are hard limits (else soft: values beyond can be typed)
    user_data: str = ""
    visible_if: str = ""  # expression on the graph's inputs (trafilatura3d.visibility)
    items: tuple = ()  # Drop down list (INT): ((value, label), ...)

    def editor_name(self) -> str:
        """The editor this parameter is shown with: its Editor if the kind has it, else the kind's default."""
        names = EDITORS.get(self.kind, ("",))
        return self.editor if self.editor in names else names[0]

    def components(self) -> int:
        """How many numbers a value of this kind holds (a vector's length, 1 for a scalar or a choice)."""
        return 4 if self.kind is ParamKind.COLOR else _VECTOR_LEN.get(self.kind, 1)

    def _clamp(self, v):
        if not self.clamp:
            return v
        if self.min is not None:
            v = max(v, self.min)
        if self.max is not None:
            v = min(v, self.max)
        return v

    def validate(self, value: Any) -> Any:
        """Return the value normalized for this kind, or raise ValueError."""
        k = self.kind
        if k is ParamKind.FLOAT and _number(value):
            return self._clamp(float(value))
        if k is ParamKind.INT and isinstance(value, int) and not isinstance(value, bool):
            return int(self._clamp(value))
        if k is ParamKind.BOOL and isinstance(value, bool):
            return value
        if k in _VECTOR_LEN and isinstance(value, (list, tuple)) and len(value) == _VECTOR_LEN[k] \
                and all(_number(x) for x in value):
            return [int(self._clamp(x)) for x in value] if k is ParamKind.INT2 else [self._clamp(float(x)) for x in value]
        if k is ParamKind.COLOR and isinstance(value, (list, tuple)) and len(value) in (3, 4) \
                and all(_number(x) for x in value):
            return [float(x) for x in value] + ([1.0] if len(value) == 3 else [])
        if k is ParamKind.ENUM and value in self.choices:
            return value
        if k is ParamKind.STRING and isinstance(value, str):
            return value
        if k is ParamKind.GRADIENT:
            g = {"keys": value} if isinstance(value, list) else value
            keys = g.get("keys") if isinstance(g, dict) else None
            if isinstance(keys, list) and keys and all(
                    isinstance(s, dict) and _number(s.get("pos")) and isinstance(s.get("color"), (list, tuple))
                    and len(s["color"]) in (3, 4) and all(_number(x) for x in s["color"]) for s in keys) \
                    and g.get("interpolation", "linear") in GRADIENT_INTERPOLATIONS \
                    and _number(g.get("midpoint", 0.5)) and _number(g.get("smoothness", 0.5)):
                return {"interpolation": g.get("interpolation", "linear"),
                        "midpoint": float(g.get("midpoint", 0.5)), "smoothness": float(g.get("smoothness", 0.5)),
                        "keys": sorted(({"pos": float(s["pos"]), "color": [float(x) for x in s["color"]]
                                         + ([1.0] if len(s["color"]) == 3 else [])} for s in keys),
                                       key=lambda s: s["pos"])}
        if k in (ParamKind.FUNCTION, ParamKind.FXMAP) and isinstance(value, dict) \
                and isinstance(value.get("nodes"), list) \
                and all(isinstance(n, dict) and "id" in n and "type" in n for n in value["nodes"]):
            return value  # typed and checked against its variables when compiled
        if k is ParamKind.CURVE and isinstance(value, list) and len(value) >= 2 and all(
                isinstance(pt, (list, tuple)) and len(pt) in (2, 6) and all(_number(x) for x in pt) for pt in value):
            return sorted(([float(x) for x in pt] for pt in value), key=lambda pt: pt[0])
        expected = f"one of {list(self.choices)}" if k is ParamKind.ENUM else k.value
        raise ValueError(f"expected {expected}, got {value!r}")


class Node(ABC):
    type: ClassVar[str]
    label: ClassVar[str]
    category: ClassVar[str] = ""
    description: ClassVar[str] = ""
    inputs: ClassVar[dict[str, Input]] = {}
    outputs: ClassVar[dict[str, Output]] = {"output": Output()}
    params: ClassVar[dict[str, Param]] = {}
    # Uniform fields the node sets itself in `encode` (e.g. a pass index), after the params.
    internal_uniforms: ClassVar[dict[str, ParamKind]] = {}
    # Storage buffers the shader reads and writes (name -> element type, e.g. "atomic<u32>"), declared as
    # `var<storage, read_write> name: array<type>` after the other bindings; `encode` passes them to the dispatch.
    buffers: ClassVar[dict[str, str]] = {}
    # Each output texel computed from the same texel of the inputs (a few texels around at most): the node can be
    # dispatched over a region of its output alone (Evaluator.recook_region; its shader adds `origin()`).
    pointwise: ClassVar[bool] = False
    # A chain of jobs of this node, each reading the previous one on this input and nobody else reading it, can be
    # computed by one dispatch (`encode_fused`, at most `max_fused` jobs): the results in between are never written
    # (Evaluator.cook).
    fuse_input: ClassVar[str] = ""
    max_fused: ClassVar[int] = 1
    # The input whose size and format "relative to input" refers to (the main input).
    main_input: ClassVar[str | None] = None
    # The id of the node's page in the public documentation it follows ("" for our own nodes).
    catalog: ClassVar[str] = ""
    # Catalog parameters deliberately not implemented: label -> reason.
    unsupported: ClassVar[dict[str, str]] = {}
    # The salt of the node's random stream on the CPU (legacy graphs, stamped noises): None = from its type id. A
    # renamed node keeps the salt of its former id, so that its look does not change.
    salt: ClassVar[int | None] = None

    def stream_salt(self) -> int:
        return sum(map(ord, self.type)) if self.salt is None else self.salt

    # --- typing -------------------------------------------------------------------------------

    # A parameter choosing the output type (a static "Color mode"), when the node has one.
    color_mode: ClassVar[str | None] = None

    def output_types(self, input_types: dict[str, PortType | None], values: dict[str, Any]) -> dict[str, PortType]:
        """Resolved output types, given what is connected to each input (its type, or None) and the
        parameter values. `any` outputs are color when the color mode says so, else when any `any`
        input is color."""
        if self.color_mode is not None:
            color = values[self.color_mode] in ("Color", True)
        else:
            color = any(t is PortType.COLOR for name, t in input_types.items()
                        if self.inputs[name].type is PortType.ANY)
        return {name: (o.type if o.type is not PortType.ANY else PortType.COLOR if color else PortType.GRAY)
                for name, o in self.outputs.items()}

    # Numbered inputs enabled by a count parameter, as "Pattern Input Number" (it
    # unlocks Pattern Input slots): {count parameter: input name prefix}. The extra inputs stay in the definition
    # (the shader reads them all) but the graph neither shows nor accepts them; reducing the count keeps
    # their connections (Multi Switch documentation: "does not remove connections").
    counted_inputs: dict[str, str] = {}

    def visible_inputs(self, values: dict[str, Any]) -> list[str]:
        hidden = set()
        for param, prefix in self.counted_inputs.items():
            count = values.get(param, self.params[param].default)
            numbered = sorted((int(m.group(1)), name) for name in self.inputs
                              if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", name)))
            hidden.update(name for k, (_, name) in enumerate(numbered) if k >= int(count))
        return [name for name in self.inputs if name not in hidden]

    def visible_outputs(self, values: dict[str, Any]) -> list[str]:
        """Outputs shown on the node (graph instances hide those whose Visible if is false)."""
        return list(self.outputs)

    def instantiate(self, values: dict[str, Any]) -> Node:
        """The definition for these parameter values: nodes whose ports depend on a parameter
        (e.g. the Pixel processor's number of inputs) return a specialized copy."""
        return self

    def extra(self, values: dict[str, Any], package) -> dict[str, Any]:
        """Data the node needs besides its values, part of the job key (e.g. a resolved file path and
        its modification time, so editing the file recooks)."""
        return {}

    def base_defaults_for(self, values: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        """Base parameter defaults that depend on the node's data (e.g. Bitmap: its image's size)."""
        own = getattr(self, "base_defaults", {})
        depth = DEPTHS.get(self.type)
        return {"output_format": {"mode": "absolute", "value": depth}, **own} if depth else own

    def primary_input(self) -> str | None:
        """The input base parameters inherit from: never a value (a 1x1 texture)."""
        if self.main_input:
            return self.main_input
        return next((k for k, i in self.inputs.items() if i.type is not PortType.VALUE), None)

    def values(self, node) -> dict[str, Any]:
        """Every parameter of a node instance, defaults filled in."""
        return {name: node.params.get(name, p.default) for name, p in self.params.items()}

    # --- cooking ------------------------------------------------------------------------------

    def encode(self, ctx: CookContext, job: Job) -> None:
        """Record the GPU work that writes the job's outputs. Default: one dispatch of `main`."""
        ctx.dispatch(self, job, job.values)

    def encode_fused(self, ctx: CookContext, chain: list[Job]) -> None:
        """Record the work writing the last job's outputs of a chain (see `fuse_input`), the same texel for texel as
        the jobs one after the other."""
        raise NotImplementedError

    # --- shader -------------------------------------------------------------------------------

    def uniform_fields(self) -> list[tuple[str, ParamKind]]:
        fields = [(n, p.kind) for n, p in self.params.items() if p.kind in _UNIFORM]
        quant = [(f"q_{o}", ParamKind.FLOAT) for o in self.outputs]
        return BUILTINS + quant + fields + list(self.internal_uniforms.items())

    def lut_params(self) -> list[str]:
        return [n for n, p in self.params.items() if p.kind in (ParamKind.GRADIENT, ParamKind.CURVE)]

    def pack(self, values: dict[str, Any]) -> bytes:
        """Values (params, built-ins, internal uniforms) as the bytes of the generated `Params` struct."""
        data = bytearray()
        for name, kind in self.uniform_fields():
            wgsl, fmt, align = _UNIFORM[kind]
            data += bytes(-len(data) % align)
            value = values[name]
            if kind is ParamKind.ENUM:
                value = self.params[name].choices.index(value)
            elif kind is ParamKind.BOOL:
                value = int(value)
            # A vec3f takes 12 bytes (aligned to 16): a following scalar fills its fourth slot.
            data += struct.pack(fmt, *value) if len(fmt) > 1 else struct.pack(fmt, value)
        return bytes(data + bytes(-len(data) % 16))

    def shader_source(self, variant: Variant) -> str:
        lines = [COMMON_WGSL, f"const WG = {WORKGROUP};"]
        lines.append("struct Params {\n" + "".join(f"    {n}: {_UNIFORM[k][0]},\n"
                                                    for n, k in self.uniform_fields()) + "};")
        lines.append("@group(0) @binding(0) var<uniform> p: Params;")
        binding = 1
        lines.append("fn size() -> vec2i { return vec2i(p.size_); }")
        lines.append("fn origin() -> vec2i { return vec2i(p.origin_); }")
        lines.append("fn tile_h() -> bool { return (p.tiling_ & 1) != 0; }")
        lines.append("fn tile_v() -> bool { return (p.tiling_ & 2) != 0; }")
        for port, (connected, gray, alpha, region, resampled) in zip(self.inputs, variant.inputs):
            lines.append(f"@group(0) @binding({binding}) var src_{port}: texture_2d<f32>;")
            binding += 1
            up = port.upper()
            lines.append(f"const {up}_CONNECTED = {'true' if connected else 'false'};")
            lines.append(f"const {up}_GRAY = {'true' if gray else 'false'};")
            swizzle = "vec4f(t.r, t.r, t.r, t.g)" if alpha else "vec4f(t.r, t.r, t.r, 1.0)" if gray else "t"
            # A region of the output's texture from origin() (Evaluator.recook_region): its size is the output's (the
            # shaders scale with it: the normal's slope) and every read goes through texel_ at the texel minus origin()
            # (the normal reads its neighbors with it).
            dims = f"vec2i(textureDimensions(src_{port}))"
            lines.append(f"fn size_{port}() -> vec2i {{ return {'vec2i(p.size_)' if region else dims}; }}")
            at = " - origin()" if region else ""
            lines.append(f"fn texel_{port}(q: vec2i) -> vec4f {{ let t = textureLoad(src_{port}, "
                         f"tile_coord(q{at}, {dims}), 0); return {swizzle}; }}")
            if region:  # same texels as the output's
                lines.append(f"fn load_{port}(c: vec2i) -> vec4f {{ return texel_{port}(c); }}")
                at = "uv * p.size_ - 0.5"
            else:
                # An input of another size is read bilinearly at the output pixel's centre (as most nodes of the
                # reference do), one of the output's size gives its texel, an unconnected one its single texel.
                read = (f"sample_{port}((vec2f(c) + 0.5) / p.size_)" if resampled
                        else f"texel_{port}(c)" if connected else f"texel_{port}(vec2i(0))")
                lines.append(f"fn load_{port}(c: vec2i) -> vec4f {{ return {read}; }}")
                at = f"uv * vec2f(size_{port}()) - 0.5"
            if variant.filtered and not region:  # the GPU's bilinear read, wrapping as tile_coord does (`smp_`)
                lines.append(f"fn sample_{port}(uv: vec2f) -> vec4f {{ "
                             f"let t = textureSampleLevel(src_{port}, smp_, uv, 0.0); return {swizzle}; }}")
            else:
                lines.append(f"fn sample_{port}(uv: vec2f) -> vec4f {{ "
                             f"let x = {at}; let i = vec2i(floor(x)); let f = fract(x); "
                             f"return mix(mix(texel_{port}(i), texel_{port}(i + vec2i(1, 0)), f.x), "
                             f"mix(texel_{port}(i + vec2i(0, 1)), texel_{port}(i + vec2i(1, 1)), f.x), f.y); }}")
            if self.inputs[port].type is PortType.VALUE:  # a value: the single texel, as is
                lines.append(f"fn value_{port}() -> vec4f {{ return textureLoad(src_{port}, vec2i(0), 0); }}")
        for i, (port, fmt) in enumerate(zip(self.outputs, variant.outputs)):
            lines.append(f"@group(0) @binding({binding}) var dst_{port}: texture_storage_2d<{fmt}, write>;")
            binding += 1
            lines.append(f"const OUT_{port.upper()}_GRAY = {'false' if fmt.startswith('rgba') else 'true'};")
            if self.outputs[port].type is PortType.VALUE:  # written once, by the invocation at (0, 0)
                lines.append(f"fn store_{port}(c: vec2i, v: vec4f) {{ if (all(c == vec2i(0))) {{ "
                             f"textureStore(dst_{port}, vec2i(0), v); }} }}")
                continue
            q = f"p.q_{port}"
            packed = "vec4f(o.r, o.a, 0.0, 0.0)" if two_channels(fmt) else "o"  # gray + alpha: (v, a)
            at = "c - origin()" if variant.regions and variant.regions[i] else "c"  # a region texture from origin()
            # a whole result written over a region (recook_region): only inside p.clip_ (x0, y0, x1, y1; none when
            # x1 is 0), where the texels around, read by the normal, were computed too
            clip = "" if at != "c" else ("if (p.clip_.z > 0.0 && (any(vec2f(c) < p.clip_.xy) || "
                                         "any(vec2f(c) >= p.clip_.zw))) { return; } ")
            quantize = "quantize_for_unorm" if "unorm" in fmt else "quantize_depth"
            lines.append(f"fn store_{port}(c: vec2i, v: vec4f) {{ {clip}var o = v; if ({q} > 0.0) {{ "
                         f"o = {quantize}(v, {q}); }} "
                         f"textureStore(dst_{port}, {at}, {packed}); }}")
        if self.outputs:
            lines.append(f"fn store(c: vec2i, v: vec4f) {{ store_{next(iter(self.outputs))}(c, v); }}")
        for name in self.lut_params():
            lines.append(f"@group(0) @binding({binding}) var lut_tex_{name}: texture_2d<f32>;")
            binding += 1
            lines.append(f"fn lut_{name}(t: f32) -> vec4f {{ let x = clamp(t, 0.0, 1.0) * {LUT_SIZE - 1}.0; "
                         f"let i = i32(floor(x)); let a = textureLoad(lut_tex_{name}, vec2i(i, 0), 0); "
                         f"let b = textureLoad(lut_tex_{name}, vec2i(min(i + 1, {LUT_SIZE - 1}), 0), 0); "
                         f"return mix(a, b, fract(x)); }}")
        if variant.filtered:
            lines.append(f"@group(0) @binding({binding}) var smp_: sampler;")
            binding += 1
        for name, element in self.buffers.items():
            lines.append(f"@group(0) @binding({binding}) var<storage, read_write> {name}: array<{element}>;")
            binding += 1
        code = self.wgsl()
        if variant.pages:  # only the pages set: elsewhere 0 written, nothing read (a pointwise node's main)
            assert self.pointwise and code.count(MAIN) == 1, self.type
            lines.append("fn page_on_(c: vec2i) -> bool {\n"
                         f"    let g = clamp(c * {PAGE_GRID} / max(size(), vec2i(1)), vec2i(0), vec2i({PAGE_GRID - 1}));\n"
                         f"    let b = u32(g.y * {PAGE_GRID} + g.x);\n"
                         "    var w = p.pages0_;\n"
                         + "".join(f"    if (b / 32u == {i}u) {{ w = p.pages{i}_; }}\n" for i in range(1, 8))
                         + "    return ((u32(w) >> (b % 32u)) & 1u) != 0u;\n}")
            code = code.replace(MAIN, MAIN + "\n    { let c_ = vec2i(id.xy) + origin(); if (!page_on_(c_)) { "
                                "if (all(c_ < size())) { store(c_, vec4f(0.0)); } return; } }")
        lines.append(code)
        return "\n".join(lines)

    def wgsl(self) -> str:
        return Path(inspect.getfile(type(self))).with_suffix(".wgsl").read_text(encoding="utf-8")


def two_channels(fmt: str) -> bool:
    """A storage format of a grayscale value and its alpha (PortType.GRAY_ALPHA): rg8unorm, rg16float…"""
    return fmt.startswith("rg") and not fmt.startswith("rgba")


@dataclass(frozen=True)
class Variant:
    """What a compiled shader depends on besides its code."""
    # per input: (connected, grayscale, with its alpha: two channels, a region of the output's texture from origin(),
    # resampled: connected and of another size than the dispatch)
    inputs: tuple[tuple[bool, bool, bool, bool, bool], ...]
    outputs: tuple[str, ...]  # storage format per output
    regions: tuple[bool, ...] = ()  # per output: a texture of the region only (written at c - origin())
    pages: bool = False  # only some pages computed (Job.extra "pages_on": the pages0_..pages7_ bits)
    # The device filters every format the inputs can have: sample_ reads through the GPU's sampler (`smp_`, bound
    # last) instead of four texel reads
    filtered: bool = False
