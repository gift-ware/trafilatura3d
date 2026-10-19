"""Shared machinery of the material, mesh based and HDRI nodes.

Material convention: a full material travels as one connection per channel. Ports are
named after the channel (`basecolor`, `normal`, `roughness`...), the same names on every material
node, so a material chains node to node by connecting same-named ports. Nodes taking several
materials suffix the port names (`basecolor_fg` / `basecolor_bg`, `basecolor_1`...). Color
channels are sRGB-encoded, normals are tangent-space maps in the node's Normal
format (OpenGL when the node has no such parameter), gray channels are linear.

Every node of the area is a `MatNode`: its shader is `lib.wgsl` + its includes + its own `.wgsl`
(with optional `//@each` ... `//@end` blocks repeated per channel). A shader binds at most 16
sampled textures (inputs + LUTs): nodes taking several materials cook one pass per channel.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import downsample, gaussian

_DIR = Path(__file__).parent
LIB = (_DIR / "lib.wgsl").read_text(encoding="utf-8")
GRAY, COLOR, ANY = PortType.GRAY, PortType.COLOR, PortType.ANY

# Channel -> (label, port type, value read when unconnected, kind: 0 color, 1 normal, 2 gray).
# Order and names follow the full material (Material Blend's channel groups).
CHANNELS = {
    "diffuse": ("Diffuse", COLOR, (0.5, 0.5, 0.5, 1.0), 0),
    "basecolor": ("Base Color", COLOR, (0.5, 0.5, 0.5, 1.0), 0),
    "normal": ("Normal", COLOR, (0.5, 0.5, 1.0, 1.0), 1),
    "specular": ("Specular", COLOR, (0.22, 0.22, 0.22, 1.0), 0),  # F0 0.04, sRGB-encoded
    "emissive": ("Emissive", COLOR, (0.0, 0.0, 0.0, 1.0), 0),
    "glossiness": ("Glossiness", GRAY, (0.5, 0.5, 0.5, 1.0), 2),
    "roughness": ("Roughness", GRAY, (0.5, 0.5, 0.5, 1.0), 2),
    "metallic": ("Metallic", GRAY, (0.0, 0.0, 0.0, 1.0), 2),
    "specular_level": ("Specular Level", GRAY, (0.5, 0.5, 0.5, 1.0), 2),
    "ambient_occlusion": ("Ambient Occlusion", GRAY, (1.0, 1.0, 1.0, 1.0), 2),
    "height": ("Height", GRAY, (0.5, 0.5, 0.5, 1.0), 2),
    "opacity": ("Opacity", GRAY, (1.0, 1.0, 1.0, 1.0), 2),
}
FULL = tuple(CHANNELS)


def material_inputs(channels=FULL, suffix: str = "", label: str = "") -> dict[str, Input]:
    """Inputs for a material: `<channel><suffix>`, labelled "<label> <Channel>"."""
    return {f"{ch}{suffix}": Input(CHANNELS[ch][1], f"{label} {CHANNELS[ch][0]}".strip(), CHANNELS[ch][2])
            for ch in channels}


def material_outputs(channels=FULL) -> dict[str, Output]:
    return {ch: Output(CHANNELS[ch][1], CHANNELS[ch][0]) for ch in channels}


def gray_in(label: str, v: float = 0.0) -> Input:
    return Input(GRAY, label, (v, v, v, 1.0))


def color_in(label: str, rgba=(0.0, 0.0, 0.0, 1.0)) -> Input:
    return Input(COLOR, label, tuple(rgba))


def expand(text: str, channels) -> str:
    """Repeat each `//@each` ... `//@end` block once per channel: `$ch` -> name, `$CH` -> upper
    name, `$K` -> kind (0 color, 1 normal, 2 gray), `$i` -> index, `$n` -> index + 1."""
    def block(m):
        body = m.group(1)
        return "".join(body.replace("$ch", ch).replace("$CH", ch.upper())
                       .replace("$K", str(CHANNELS[ch][3] if ch in CHANNELS else 2))
                       .replace("$n", str(i + 1)).replace("$i", str(i)) for i, ch in enumerate(channels))
    return re.sub(r"//@each\n(.*?)//@end\n", block, text, flags=re.S)


class MatNode(Node):
    """A node of this area: shader = lib.wgsl + its file (channel blocks expanded over `channels`)."""

    channels: tuple[str, ...] = FULL
    wgsl_file: str | None = None  # another stem than the module's
    includes: tuple[str, ...] = ()  # shared .wgsl files next to the module, prepended

    def source(self) -> Path:
        return Path(inspect.getfile(type(self)))

    def wgsl(self) -> str:
        path = self.source()
        parts = [LIB, self.channel_outs()] + [(path.parent / f).read_text(encoding="utf-8") for f in self.includes]
        path = path.with_name(self.wgsl_file) if self.wgsl_file else path.with_suffix(".wgsl")
        return "\n".join(parts + [expand(path.read_text(encoding="utf-8"), self.channels)])

    def channel_outs(self) -> str:
        """`out_<ch>(c, v)` for every channel output: what to store at `c` for the value `v`. With the
        channel's toggle (`channel_toggles`) off, the same-named input unchanged, or the channel's
        default value when the node has no such input; without a toggle, `v`."""
        lines = []
        for ch, (_, _, default, _) in CHANNELS.items():
            if ch not in self.outputs:
                continue
            body = "v"
            if f"channel_{ch}" in self.params:
                off = f"load_{ch}(c)" if ch in self.inputs else f"vec4f{tuple(default)}"
                body = f"select({off}, v, p.channel_{ch} != 0)"
            lines.append(f"fn out_{ch}(c: vec2i, v: vec4f) -> vec4f {{ return {body}; }}")
        return "\n".join(lines)

class Helper(MatNode):
    """A pass of a multi-pass node (not in the library). Ports with the job's port names are bound
    to the job's inputs automatically."""
    registered = False


class Pass(Helper):
    """A pass of `parent` with its own shader file (next to the parent's module), ports and
    internal uniforms; parameters, channels and includes default to the parent's."""

    def __init__(self, parent: MatNode, name: str, wgsl_file: str, inputs: dict, outputs: dict,
                 internal: dict | None = None, params: dict | None = None, channels=None, includes=None,
                 channel: str | None = None, subst: dict | None = None):
        self.type = f"materials:{parent.type}:{name}"
        self.label = f"{parent.label} ({name})"
        self.inputs = inputs
        self.outputs = outputs
        self.params = parent.params if params is None else params
        self.internal_uniforms = internal or {}
        self.wgsl_file = wgsl_file
        self.channels = parent.channels if channels is None else channels
        self.includes = parent.includes if includes is None else includes
        self._source = parent.source()
        self.channel = channel  # a per-channel pass: `$ch`, `$CH`, `$K` replaced in the whole file
        self.subst = subst or {}  # other replacements, applied first

    def source(self) -> Path:
        return self._source

    def wgsl(self) -> str:
        text = super().wgsl()
        for k in sorted(self.subst, key=len, reverse=True):
            text = text.replace(k, self.subst[k])
        if self.channel:
            ch = self.channel
            text = (text.replace("$ch", ch).replace("$CH", ch.upper())
                    .replace("$K", str(CHANNELS[ch][3] if ch in CHANNELS else 2)))
        return text


def channel_passes(node: MatNode, make) -> dict:
    """One pass per channel of the node, made once by `make(node, channel)`."""
    return passes_of(node, lambda n: {ch: make(n, ch) for ch in n.channels}, "channels")


# --- reusable passes ----------------------------------------------------------------------------

def with_tiling(job, tiling: int):
    """The job seen with another tiling (bits 1 H, 2 V), for passes that must wrap or clamp."""
    return dataclasses.replace(job, tiling=tiling)


def mean(ctx, job, tex, gray: bool):
    """A 1x1 texture holding the average of `tex`."""
    while tex.width > 1 or tex.height > 1:
        tex = downsample(ctx, job, tex, gray, (min(tex.width, 16), min(tex.height, 16)))
    return tex


def input_texture(ctx, job, port: str):
    """(texture, is_gray) of what is connected to `port`, or None."""
    src = job.inputs.get(port)
    if not src:
        return None
    j, out = src
    return ctx.ev.texture(j, out), j.is_gray(out)


def px256(job) -> float:
    """Pixels per "pixel at 256 px" of the job's output (resolution independence)."""
    return job.size[0] / 256.0


def log2i(x: int) -> int:
    return max(0, int(math.log2(max(1, x))))


# --- parameter shorthands -----------------------------------------------------------------------

def F(default, label, lo=0.0, hi=1.0, group=""):
    return Param(ParamKind.FLOAT, float(default), label, lo, hi, group=group)


def I(default, label, lo=None, hi=None, group=""):
    return Param(ParamKind.INT, int(default), label, lo, hi, group=group)


def B(default, label, group=""):
    return Param(ParamKind.BOOL, bool(default), label, group=group)


def E(default, label, choices, group=""):
    return Param(ParamKind.ENUM, default, label, choices=tuple(choices), group=group)


def C(default, label, group=""):
    return Param(ParamKind.COLOR, list(default) + ([1.0] if len(default) == 3 else []), label, group=group)


def F2(default, label, lo=0.0, hi=1.0, group=""):
    return Param(ParamKind.FLOAT2, [float(x) for x in default], label, lo, hi, group=group)


def F3(default, label, lo=-1.0, hi=1.0, group=""):
    return Param(ParamKind.FLOAT3, [float(x) for x in default], label, lo, hi, group=group)


def F4(default, label, lo=-2.0, hi=2.0, group=""):
    return Param(ParamKind.FLOAT4, [float(x) for x in default], label, lo, hi, group=group)


def I2(default, label, lo=None, hi=None, group=""):
    return Param(ParamKind.INT2, [int(x) for x in default], label, lo, hi, group=group)


def channel_toggles(on, channels=FULL) -> dict:
    """The Channels group: one switch per channel (`channel_<ch>`, the `on` ones on), read by
    `MatNode.channel_outs`: a channel switched off is passed through (or set to its default value)."""
    return {f"channel_{ch}": B(ch in on, CHANNELS[ch][0], group="Channels") for ch in channels}


MATRIX = [1.0, 0.0, 0.0, 1.0]  # the transformation matrix (2x2, column major like Transformation 2D)
NORMAL_FORMATS = ("DirectX", "OpenGL")


_PASSES: dict = {}


def passes_of(node: MatNode, make, tag: str = ""):
    """The node's passes (one set per `tag`), made once by `make(node)`."""
    key = (node.type, tag)
    if key not in _PASSES:
        _PASSES[key] = make(node)
    return _PASSES[key]


class PreBlurNode(MatNode):
    """Three steps: `<stem>_pre.wgsl` computes a grayscale map (output `pre`), a gaussian blur of
    `blur_sigma(job)` pixels, then `<stem>.wgsl` with the blurred map as input `blurred` (read it with
    `sample_blurred`)."""

    pre_inputs: tuple[str, ...] = ()  # the inputs `<stem>_pre.wgsl` reads (and the final pass doesn't,
    final_also: tuple[str, ...] = ()  # ... except these)
    pre_wgsl: str = ""  # a pre pass shared by several nodes, instead of `<stem>_pre.wgsl`

    def blur_sigma(self, job) -> float:
        raise NotImplementedError

    def encode(self, ctx, job):
        stem = self.source().stem
        pre, final = passes_of(self, lambda n: (
            Pass(n, "pre", n.pre_wgsl or f"{stem}_pre.wgsl", {k: n.inputs[k] for k in n.pre_inputs}, {"pre": Output(GRAY)}),
            Pass(n, "final", f"{stem}.wgsl", {**{k: v for k, v in n.inputs.items()
                                                 if k not in n.pre_inputs or k in n.final_also},
                                              "blurred": Input(GRAY)}, n.outputs)))
        t = ctx.scratch(job, True)
        ctx.dispatch(pre, job, job.values, dsts={"pre": t})
        b = gaussian(ctx, job, t, True, self.blur_sigma(job))
        ctx.dispatch(final, job, job.values, srcs={"blurred": (b, True)})
