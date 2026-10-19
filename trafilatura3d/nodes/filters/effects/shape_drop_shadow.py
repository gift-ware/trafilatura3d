import math

from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ..basic._lib import blur2d
from ._fx import Effect, scale, source
from .shape_stroke import SHAPE_OUTPUTS, SHAPE_PARAMS

# The shadow's softness: a gaussian of sigma SOFTNESS x Size^2, in pixels at 256 (fitted on the node's output).
SOFTNESS = 29.0
TAPS = 8  # bilinear taps per side: a soft shadow needs few


class ShapeDropShadow(Effect):
    """Drop shadow under a shape: a black and white mask (shown in Mask Color) or a color image's alpha. The
    shape's coverage blurred (a gaussian that grows with the square of Size), read Distance / 2 of the image
    away from it in the direction opposite to Angle (turns: 0.25 = down), choked by Spread (coverage above
    1 - Spread is full) and scaled by Opacity, laid under the shape in (Shadow) Color. Output: the composite
    with its transparency, optionally premultiplied; Mask: the shadow alone, outside the shape."""

    type = "shape_drop_shadow"
    label = "Shape Drop Shadow"
    category = "Filters/Effects"
    catalog = "shape_drop_shadow"
    libs = ("_shape.wgsl",)
    inputs = {"input": Input(label="Input")}
    outputs = SHAPE_OUTPUTS
    params = {
        "angle": Param(ParamKind.FLOAT, 0.25, "Angle", 0.0, 1.0),
        "distance": Param(ParamKind.FLOAT, 0.02, "Distance", -0.5, 0.5),
        "size": Param(ParamKind.FLOAT, 0.15, "Size", 0.0, 1.0),
        "spread": Param(ParamKind.FLOAT, 0.0, "Spread", 0.0, 1.0),
        "opacity": Param(ParamKind.FLOAT, 0.5, "Opacity", 0.0, 1.0),
        "shadow_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "(Shadow) Color"),
        **SHAPE_PARAMS,
    }

    def encode(self, ctx, job):
        v = job.values
        src = source(ctx, job, "input")
        blur = blur2d(ctx, job, src, src[1], SOFTNESS * v["size"] ** 2 * scale(job), taps=TAPS)
        a = v["angle"] * 2.0 * math.pi
        offset = [0.5 * v["distance"] * math.cos(a), -0.5 * v["distance"] * math.sin(a)]
        ctx.dispatch(FINAL, job, {**v, "offset": offset}, srcs={"blur": (blur, src[1])})


class _Final(Effect):
    type = "fx_shape_drop_shadow_final"
    registered = False
    libs = ("_shape.wgsl",)
    inputs = {"input": Input(), "blur": Input()}
    outputs = ShapeDropShadow.outputs
    params = ShapeDropShadow.params
    internal_uniforms = {"offset": ParamKind.FLOAT2}


FINAL = _Final()
