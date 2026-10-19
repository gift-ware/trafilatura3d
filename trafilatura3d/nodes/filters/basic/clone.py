from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BOX, BasicNode, blur2d, helper, scale, source


class Clone(BasicNode):
    """Crude clone stamp: Source moved by Offset (positive: left and up; wrapping on
    tiled axes, transparent elsewhere) is copied over the original where Mask is white, straight copy
    weighted by the copy's alpha. The mask (black when unconnected: no effect) stays in place, softened
    by Blur Mask (the atomic Blur: a box of that half-width in pixels at 256 px)."""

    type = "clone"
    label = "Clone (Filter Node)"
    category = "Filters/Transforms"
    catalog = "clone"
    inputs = {"source": Input(PortType.COLOR, "Source"), "mask": Input(PortType.GRAY, "Mask")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Offset", -1.0, 1.0),
        "blur_mask": Param(ParamKind.FLOAT, 0.5, "Blur Mask", 0.0, 10.0),
    }

    def encode(self, ctx, job):
        mask = source(ctx, job, "mask")
        r = job.values["blur_mask"] * scale(job)
        if r > 0.05:
            mask = (blur2d(ctx, job, mask, True, r, BOX), True)
        ctx.dispatch(FINAL, job, job.values, srcs={"mask": mask})


FINAL = helper("clone_final", "clone.wgsl", {"source": Input(PortType.COLOR), "mask": Input(PortType.GRAY)},
               {"output": Clone.outputs["output"]}, params=Clone.params)
