import math

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import blur2d

from ._fx import Effect, helper, nearest_seeds, scale, source
from .shape_stroke import SHAPE_OUTPUTS, SHAPE_PARAMS


class ShapeGlow(Effect):
    """Soft glow around a shape: a black and white mask (shown in Mask Color) or a color image's alpha.
    The shape (alpha above 2%, for Width < 0 what is below 98%) spreads a linear falloff (exact distance, jump
    flooding) softened by a gaussian. Soft: falloff over 8 x |Width| pixels at 256, sigma 28.8 x Width^2, then a
    gamma 0.44 that brightens the tail; Precise: falloff over 128 x Width^2, sigma 0.9 x (1 - |Width|) (sigmas
    and gamma fitted on the output). Spread divides the glow by 1 - Spread (solid near the edge before it
    fades). Width > 0 glows outside (under the shape), < 0 inside (over it); Opacity and (Shadow)
    Color tint it. Output: the composite with its transparency, optionally premultiplied; Mask: the
    glow alone, outside the shape (inside it for Width < 0)."""

    type = "shape_glow"
    label = "Shape Glow"
    category = "Filters/Effects"
    catalog = "shape_glow"
    libs = ("_shape.wgsl",)
    inputs = {"input": Input(label="Input")}
    outputs = SHAPE_OUTPUTS
    params = {
        "mode": Param(ParamKind.ENUM, "Soft", "Mode", choices=("Soft", "Precise")),
        "width": Param(ParamKind.FLOAT, 0.25, "Width", -1.0, 1.0),
        "spread": Param(ParamKind.FLOAT, 0.0, "Spread", 0.0, 1.0),
        "opacity": Param(ParamKind.FLOAT, 1.0, "Opacity", 0.0, 1.0),
        "glow_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "(Shadow) Color"),
        **SHAPE_PARAMS,
    }

    def encode(self, ctx, job):
        v = job.values
        outer = v["width"] >= 0.0
        src = source(ctx, job, "input")
        # The shape: any visible alpha (above 2%), or for an inner glow anything not fully opaque around it.
        seeds = nearest_seeds(ctx, job, src, 0.02 if outer else 0.98, channel=0 if src[1] else 1)
        w = abs(v["width"])
        soft = v["mode"] == "Soft"
        reach = 8.0 * w * scale(job) if soft else 0.5 * w * w * job.size[0]
        falloff = ctx.scratch(job, True)
        ctx.dispatch(FALLOFF, job, {"reach": reach, "outer": int(outer)}, srcs={"seeds": (seeds, False)},
                     dsts={"out": falloff})
        # Gaussian sigma in pixels at 256 (fitted on the output): wide and growing fast in Soft, a light
        # antialiasing that vanishes at |Width| 1 in Precise.
        sigma = (SOFT_SIGMA * w * w if soft else PRECISE_SIGMA * (1.0 - w)) * scale(job)
        if sigma > 0.3:  # taps 1.5 sigma per side are enough for this smooth ramp
            falloff = blur2d(ctx, job, (falloff, True), True, sigma, taps=max(4, math.ceil(1.5 * sigma)))
        ctx.dispatch(FINAL, job, v, srcs={"blur": (falloff, True)})


SOFT_SIGMA, PRECISE_SIGMA = 28.8, 0.9


FALLOFF = helper("fx_shape_falloff", "_shape_falloff.wgsl", {"seeds": Input(PortType.COLOR)},
                 {"out": Output(PortType.GRAY)}, {"reach": ParamKind.FLOAT, "outer": ParamKind.INT},
                 libs=("_shape_sd.wgsl",))


class _Final(Effect):
    type = "fx_shape_glow_final"
    registered = False
    libs = ("_shape.wgsl",)
    inputs = {"input": Input(), "blur": Input(PortType.GRAY)}
    outputs = ShapeGlow.outputs
    params = ShapeGlow.params


FINAL = _Final()
