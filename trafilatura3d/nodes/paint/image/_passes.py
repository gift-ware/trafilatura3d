"""Shared passes of the image filters: helper nodes running this folder's WGSL, blurs of scratch textures, the
wrap switch of the distance based filters and the source (default noise, previous input or custom noise) of
Warp and Blur Slope."""

from __future__ import annotations

from dataclasses import replace

from trafilatura3d.nodes.base import Input, Node, Output, ParamKind, PortType
from trafilatura3d.nodes.filters.effects._fx import gauss
from trafilatura3d.nodes.filters.basic._lib import blur2d
from trafilatura3d.nodes.paint._lib import DIR, LIB, MAIN

HERE = DIR / "image"
SOURCES = ("Default Noise", "Previous Input", "Custom Noise")
DIVIDERS = ("1", "10", "100")


def helper(type_: str, shader: str, inputs: dict, outputs: dict, params: dict | None = None,
           internal: dict | None = None, main: bool = False, includes: tuple = ("_image",)) -> Node:
    """A pass (not in the library) running `lib.wgsl` + `includes` + `shader` of this folder (+ the `main` that
    stores `effect` when `main`)."""

    def wgsl(self):
        parts = [LIB, *((HERE / f"{i}.wgsl").read_text(encoding="utf-8") for i in includes), (HERE / shader).read_text(encoding="utf-8")]
        return "\n".join(parts + ([MAIN] if main else []))

    cls = type(f"_{type_}", (Node,), {"type": type_, "registered": False, "inputs": inputs, "outputs": outputs,
                                      "params": params or {}, "internal_uniforms": internal or {}, "wgsl": wgsl})
    return cls()


def px(job) -> float:
    """Output pixels per pixel of a 256 px image."""
    return job.size[0] / 256.0


def wrapping(job, wrap: bool):
    """The job as its passes see it: without tiling when `wrap` is off (distances and blurs stop at the
    borders)."""
    return job if wrap else replace(job, tiling=0)


def blur_to(ctx, job, tex, gray: bool, radius256: float, wrap: bool = True, dst=None):
    """Gaussian blur of the scratch `tex` (radius in pixels at 256) into `dst` (None: the job's output)."""
    return blur2d(ctx, wrapping(job, wrap), (tex, gray), gray, float(radius256) * px(job),
                  dst=dst if dst is not None else ctx.ev.texture(job, "output"))


SOURCE = helper("paint_image_source", "_source.wgsl",
                {"input": Input(), "custom_noise": Input(PortType.GRAY, default=(0.5, 0.5, 0.5, 1.0))},
                {"out": Output(PortType.GRAY)},
                internal={"s_mode": ParamKind.INT, "s_tiling": ParamKind.FLOAT, "s_balance": ParamKind.FLOAT,
                          "s_contrast": ParamKind.FLOAT})


def make_source(ctx, job, mode: str, blur256: float, balance: float, contrast: float, tiling: float):
    """The grayscale source of Warp and Blur Slope (a scratch texture): the default noise (seeded by the node's
    Random Seed), the input's luminance or the Custom Noise input, repeated `tiling` times, balanced, contrasted
    and blurred (pixels at 256)."""
    s = ctx.scratch(job, True)
    ctx.dispatch(SOURCE, job, {"s_mode": SOURCES.index(mode), "s_tiling": float(tiling), "s_balance": float(balance),
                               "s_contrast": float(contrast)}, dsts={"out": s})
    if blur256 > 0.0:
        s = gauss(ctx, job, (s, True), blur256 * px(job))
    return s
