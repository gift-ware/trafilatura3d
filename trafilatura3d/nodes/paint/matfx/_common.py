"""Shared pieces of the MatFX filters and Stylization (material filters, conventions of paint/_lib.py).

Each node reads the material channels it changes (ports named after the channels, materials/_base.py) and
writes them back under the same names, keeping each channel's coverage (alpha). Colors are sRGB-encoded, normals
tangent space OpenGL, and height is changed relatively (the Paint height is 0 when flat, the Graph's 0.5).

A node's `.wgsl` defines `fn fx(c: vec2i)` that stores every output; passes (`stage`) have their own file
`<node>_<pass>.wgsl` in this folder defining the same `fx`. Both get paint/lib.wgsl and `common.wgsl`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from trafilatura3d.nodes.base import Node, Param, ParamKind
from trafilatura3d.nodes.paint._lib import FILTERS, LIB, PaintNode

HERE = Path(__file__).parent
COMMON = (HERE / "common.wgsl").read_text(encoding="utf-8")
MAIN = """
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    fx(c);
}"""
MATFX = FILTERS + "/MatFX"
ADVANCED = FILTERS + "/Advanced"


def F(label, default, lo=0.0, hi=1.0, group=""):
    return Param(ParamKind.FLOAT, float(default), label, lo, hi, group=group)


def I(label, default, lo=None, hi=None, group=""):  # noqa: E743
    return Param(ParamKind.INT, int(default), label, lo, hi, group=group)


def B(label, default=False, group=""):
    return Param(ParamKind.BOOL, bool(default), label, group=group)


def E(label, choices, default=None, group=""):
    return Param(ParamKind.ENUM, default or choices[0], label, choices=tuple(choices), group=group)


def C(label, rgb, group=""):
    return Param(ParamKind.COLOR, [float(x) for x in rgb] + [1.0], label, group=group, editor="Color")


class MatFX(PaintNode):
    """A material filter of this folder: shader = paint lib + common.wgsl + its own `.wgsl` + `main`."""

    category = MATFX

    def wgsl(self) -> str:
        own = Path(inspect.getfile(type(self))).with_suffix(".wgsl").read_text(encoding="utf-8")
        return "\n".join([LIB, COMMON, own, MAIN])


def stage(type_: str, shader: str | tuple, inputs: dict, outputs: dict, params: dict | None = None,
          internal: dict | None = None) -> Node:
    """A pass with its own shader file of this folder (not in the library); a tuple of files: shared code first."""
    files = (shader,) if isinstance(shader, str) else shader

    def wgsl(self):
        return "\n".join([LIB, COMMON, *((HERE / f).read_text(encoding="utf-8") for f in files), MAIN])

    cls = type(f"_{type_}", (Node,), {"type": type_, "registered": False, "inputs": inputs, "outputs": outputs,
                                      "params": params or {}, "internal_uniforms": internal or {}, "wgsl": wgsl})
    return cls()

