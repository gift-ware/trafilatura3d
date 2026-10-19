from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.filters.basic._lib import gaussian
from trafilatura3d.nodes.materials._base import (COLOR, GRAY, MatNode, Pass, color_in, gray_in, input_texture, mean,
                                           passes_of, px256, with_tiling)
from trafilatura3d.nodes.materials.scan._common import CATEGORY
from trafilatura3d.nodes.materials.scan._equalize import equalizer_params

MASK_BLUR = 0.65  # the mask's Blur as a gaussian sigma, in pixels at 256 px per unit


def equalize_sources(ctx, job, node: MatNode, port: str) -> dict:
    """The blurred (Radius) and averaged copies of an input and its mask (Mask Source, then its Blur):
    the sources of the final pass. `node`'s `mask_<port>` pass computes the mask."""
    t = input_texture(ctx, job, port)
    if t is None:
        z = ctx.ev.constant((0.0, 0.0, 0.0, 1.0))
        return {"blurred": (z, False), "avg": (z, False), "m": (z, True)}
    v = job.values
    tj = with_tiling(job, 3 if v["input_tiled"] else 0)
    avg = mean(ctx, job, t[0], t[1])
    m = ctx.scratch(job, True)
    ctx.dispatch(passes_of(node, mask_pass(port), port), job, v, srcs={port: t, "avg": (avg, t[1])}, dsts={"mask": m})
    if v["blur"] > 0.0:
        m = gaussian(ctx, job, m, True, v["blur"] * MASK_BLUR * px256(job))
    return {"blurred": (gaussian(ctx, tj, t[0], t[1], v["radius"] * px256(job)), t[1]), "avg": (avg, t[1]),
            "m": (m, True)}


def mask_pass(port: str):
    """The pass computing the mask of input `port`."""
    return lambda n: Pass(n, f"mask_{port}", "equalize_mask.wgsl",
                          {port: n.inputs[port], "mask_input": n.inputs["mask_input"], "avg": Input(COLOR)},
                          {"mask": Output(GRAY)}, subst={"$in": port})


class ColorEqualizer(MatNode):
    """A high-pass for color: evens out color differences larger than Radius by replacing each
    pixel's neighbourhood color with the image average (or a custom Hue / Chroma / Luma color),
    keeping the detail and the overall color. Bright/Dark Balance sets how much darker tints are
    removed. The effect can be limited to a mask: the pixels near the image average or near Color
    in hue, chroma and luma (within the ranges), or the Mask Input; the mask, softened by Blur, is
    the Mask output."""

    type = "color_equalizer"
    label = "Color Equalizer"
    category = CATEGORY
    catalog = "color_equalizer"
    includes = ("_equalize.wgsl",)
    inputs = {"input": color_in("Input", (0.5, 0.5, 0.5, 1.0)), "mask_input": gray_in("Mask Input", 1.0)}
    outputs = {"output": Output(COLOR, "Output"), "mask": Output(GRAY, "Mask")}
    params = equalizer_params()

    def encode(self, ctx, job):
        main = passes_of(self, lambda n: Pass(n, "main", "color_equalizer.wgsl",
                                              {"input": n.inputs["input"], "blurred": Input(COLOR),
                                               "avg": Input(COLOR), "m": Input(GRAY)}, n.outputs))
        ctx.dispatch(main, job, job.values, srcs=equalize_sources(ctx, job, self, "input"))
