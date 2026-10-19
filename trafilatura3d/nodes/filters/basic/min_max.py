from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode, helper, reduce, source, source_size


class MinMax(BasicNode):
    """The darkest and brightest values of a grayscale image (GPU reduction), as two values."""

    type = "min_max"
    label = "Min Max"
    category = "Filters/Adjustments"
    catalog = "min_max"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"min": Output(PortType.VALUE, "Min"), "max": Output(PortType.VALUE, "Max")}

    def encode(self, ctx, job):
        lo, hi, _ = reduce(ctx, job, source(ctx, job, "input"), source_size(job, "input"))
        ctx.dispatch(FINAL, job, job.values, srcs={"lo": (lo, False), "hi": (hi, False)})


FINAL = helper("min_max_final", "min_max.wgsl", {"lo": Input(PortType.COLOR), "hi": Input(PortType.COLOR)},
               {"min": Output(PortType.VALUE), "max": Output(PortType.VALUE)})
