from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.filters.effects._fx import nearest_seeds, source
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, E, F, Filter

from ._passes import blur_to, helper

POSITIONS = ("Outside", "Inside", "Both")


class MaskOutline(Filter):
    """An outline along the edge of the mask (the input cut at Threshold): a band Width wide (pixels at 256)
    outside the edge, inside it, or on both sides (Mask Position), white at the edge fading to black at Width
    (Curve Shape: 0 a solid band, 0.5 linear, 1 a thin line), blurred by Blur (pixels at 256). Start From Black
    inverts it."""
    type = "paint_mask_outline"
    label = "Mask Outline"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.mask_outline"
    doc = DOC + "filters/advanced-filters/mask-outline"
    params = {"mask_position": E("Mask Position", POSITIONS), "start_from_black": B("Start From Black", False),
              "threshold": F("Threshold", 0.5, 0.0, 1.0), "width": F("Width", 4.0, 0.0, 64.0),
              "blur": F("Blur", 0.5, 0.0, 16.0), "curve_shape": F("Curve Shape", 0.25, 0.0, 1.0)}

    def encode(self, ctx, job):
        v = job.values
        seeds = nearest_seeds(ctx, job, source(ctx, job, "input"), v["threshold"])
        if v["blur"] <= 0.0:
            ctx.dispatch(BAND, job, v, srcs={"seeds": (seeds, False)})
            return
        gray = job.is_gray("output")
        band = ctx.scratch(job, gray)
        ctx.dispatch(BAND, job, v, srcs={"seeds": (seeds, False)}, dsts={"output": band})
        blur_to(ctx, job, band, gray, v["blur"])


BAND = helper("paint_mask_outline_band", "mask_outline_band.wgsl", {"input": Input(), "seeds": Input(PortType.COLOR)},
              {"output": Output()}, MaskOutline.params, main=True, includes=("_image", "_sd"))
