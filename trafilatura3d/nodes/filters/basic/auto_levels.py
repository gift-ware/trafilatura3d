from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode, helper, reduce, source, source_size


class AutoLevels(BasicNode):
    """Linear remap of the image's darkest value to 0 and brightest to 1. The minimum and maximum
    are computed on the GPU (reduction); a colour image uses the min and max over R, G, B, so hues
    are kept. A uniform image passes through unchanged."""

    type = "auto_levels"
    label = "Auto Levels"
    category = "Filters/Adjustments"
    catalog = "auto_levels"
    inputs = {"input": Input(label="Input")}

    def encode(self, ctx, job):
        lo, hi, _ = reduce(ctx, job, source(ctx, job, "input"), source_size(job, "input"))
        ctx.dispatch(FINAL, job, job.values, srcs={"lo": (lo, False), "hi": (hi, False)})


FINAL = helper("auto_levels_final", "auto_levels.wgsl",
               {"input": Input(), "lo": Input(PortType.COLOR), "hi": Input(PortType.COLOR)}, {"output": Output()})
