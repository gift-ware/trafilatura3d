import math

from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.filters.effects._fx import gauss, source
from trafilatura3d.nodes.paint._lib import DOC, C, F, Filter

from ._passes import helper, px


class DropShadow(Filter):
    """Drop shadow of the painted shape (the input's alpha; a grayscale input is a mask: its value): the shape
    moved by Distance (pixels at 256) towards Angle (turns, clockwise from +X: 0.125 = down right), blurred by Size
    (pixels at 256), choked by Spread, laid under the input in Color at Opacity. On a mask the shadow adds to the
    mask (Color applies to color inputs)."""
    type = "paint_drop_shadow"
    label = "Drop Shadow"
    catalog = "painter.filter.drop_shadow"
    doc = DOC + "filters/standard-filters/drop-shadow"
    params = {"angle": F("Angle", 0.125, 0.0, 1.0), "distance": F("Distance", 8.0, 0.0, 64.0),
              "size": F("Size", 6.0, 0.0, 64.0), "spread": F("Spread", 0.0, 0.0, 1.0),
              "opacity": F("Opacity", 0.75, 0.0, 1.0), "color": C("Color", (0.0, 0.0, 0.0))}

    def encode(self, ctx, job):
        v = job.values
        src = source(ctx, job, "input")
        a = v["angle"] * 2.0 * math.pi
        d = v["distance"] / 256.0
        shift = (-math.cos(a) * d, -math.sin(a) * d)  # the shadow at p is the shape at p - direction x distance
        shadow = gauss(ctx, job, src, v["size"] * px(job), channel=0 if src[1] else 1, shift=shift)
        ctx.dispatch(FINAL, job, v, srcs={"shadow": (shadow, True)})


FINAL = helper("paint_drop_shadow_final", "drop_shadow_final.wgsl", {"input": Input(), "shadow": Input(PortType.GRAY)},
               {"output": Output()}, DropShadow.params, main=True, includes=("_image", "_behind"))
