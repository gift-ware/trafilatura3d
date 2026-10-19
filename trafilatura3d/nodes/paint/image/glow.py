from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.filters.effects._fx import gauss, nearest_seeds, source
from trafilatura3d.nodes.paint._lib import DOC, B, C, E, F, Filter

from ._passes import helper, px


class Glow(Filter):
    """A glow spreading outwards from what the input shows (its alpha; with Used on a Mask, or on a grayscale input,
    its value): Soft = a gaussian blur of the shape over Width (pixels at 256), Precise = a linear falloff over
    Width from the shape's edge (distance by jump flooding, shape cut at 0.5). Spread chokes it, Opacity scales
    it; laid under the input in Color, or added to the mask's value (Color applies to color inputs)."""
    type = "paint_glow"
    label = "Glow"
    catalog = "painter.filter.glow"
    doc = DOC + "filters/standard-filters/glow"
    params = {"used_on_a_mask": B("Used on a Mask", False), "mode": E("Mode", ("Soft", "Precise")),
              "width": F("Width", 16.0, 0.0, 64.0), "spread": F("Spread", 0.0, 0.0, 1.0),
              "opacity": F("Opacity", 1.0, 0.0, 1.0), "color": C("Color", (1.0, 0.85, 0.4))}

    def encode(self, ctx, job):
        v = job.values
        src = source(ctx, job, "input")
        mask = bool(v["used_on_a_mask"]) or src[1]
        if v["mode"] == "Soft":
            glow = gauss(ctx, job, src, v["width"] * px(job), channel=2 if mask else 1)
        else:
            seeds = nearest_seeds(ctx, job, src, 0.5, 0 if mask else 1)
            glow = ctx.scratch(job, True)
            ctx.dispatch(FALLOFF, job, v, srcs={"seeds": (seeds, False)}, dsts={"out": glow})
        ctx.dispatch(FINAL, job, {**v, "mask_": int(mask)}, srcs={"glow": (glow, True)})


FALLOFF = helper("paint_glow_falloff", "glow_falloff.wgsl", {"seeds": Input(PortType.COLOR)},
                 {"out": Output(PortType.GRAY)}, Glow.params, includes=("_image", "_sd"))
FINAL = helper("paint_glow_final", "glow_final.wgsl", {"input": Input(), "glow": Input(PortType.GRAY)},
               {"output": Output()}, Glow.params, {"mask_": ParamKind.INT}, main=True, includes=("_image", "_behind"))
