from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.filters.effects._fx import gauss, nearest_seeds, source
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, E, F, Filter

from ._passes import helper, px, wrapping


class BevelSmooth(Filter):
    """A gradient from the borders of the mask (the input cut at Mask Threshold, after a Smoothing blur in pixels
    at 256 that rounds its corners): Outward from 1 at the border to 0 at Distance (pixels at 256) outside, the
    mask staying 1; Inward from 0 at the border to 1 at Distance inside; Both from 0 at Distance outside to 1 at
    Distance inside. Curve Offset moves the borders outward (> 0) or inward by a fraction of Distance; Curve Shape
    bends the ramp (Rounded: a quarter circle, Pinched: its mirror). The Distance Map scales Distance per pixel
    by Distance Map Multiplier (0: no effect). Bevel Wrap measures across the texture borders."""
    type = "paint_bevel_smooth"
    label = "Bevel Smooth"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.bevel_smooth"
    doc = DOC + "filters/advanced-filters/bevel-smooth"
    inputs = {"input": Input(label="Input"),
              "distance_map": Input(PortType.GRAY, "Distance Map", (1.0, 1.0, 1.0, 1.0))}
    params = {"direction": E("Direction", ("Inward", "Outward", "Both")), "distance": F("Distance", 20.0, 0.0, 64.0),
              "smoothing": F("Smoothing", 1.0, 0.0, 16.0), "curve_offset": F("Curve Offset", 0.0, -1.0, 1.0),
              "curve_shape": E("Curve Shape", ("Rounded", "Pinched")),
              "mask_threshold": F("Mask Threshold", 0.5, 0.0, 1.0),
              "distance_map_multiplier": F("Distance Map Multiplier", 1.0, 0.0, 1.0),
              "bevel_wrap": B("Bevel Wrap", True)}

    def encode(self, ctx, job):
        v = job.values
        j = wrapping(job, v["bevel_wrap"])
        mask = source(ctx, job, "input")
        if v["smoothing"] > 0.0:
            mask = (gauss(ctx, j, mask, v["smoothing"] * px(job), channel=2), True)
        seeds = nearest_seeds(ctx, j, mask, v["mask_threshold"])
        ctx.dispatch(FINAL, j, v, srcs={"seeds": (seeds, False)})


FINAL = helper("paint_bevel_smooth_final", "bevel_smooth_final.wgsl",
               {"input": Input(), "distance_map": Input(PortType.GRAY, default=(1.0, 1.0, 1.0, 1.0)),
                "seeds": Input(PortType.COLOR)},
               {"output": Output()}, BevelSmooth.params, main=True, includes=("_image", "_sd"))
