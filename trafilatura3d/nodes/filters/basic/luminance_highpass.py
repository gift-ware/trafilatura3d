import math

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode, blur2d, helper, reduce, scale, source, source_size


class LuminanceHighpass(BasicNode):
    """Highpass on the luminance only: the slow changes of the luminance (Rec. 709) are flattened to its mean by
    adding the same amount to every channel, mean - local luminance (a gaussian of sigma Radius / sqrt 3, pixels at
    256 px), so the colour differences stay as they are. Clipped to 0..1 once, at the end."""

    type = "luminance_highpass"
    label = "Luminance Highpass"
    category = "Filters/Adjustments"
    catalog = "luminance_highpass"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"radius": Param(ParamKind.FLOAT, 6.0, "Radius", 0.0, 64.0)}

    def encode(self, ctx, job):
        src = source(ctx, job, "input")
        blurred = blur2d(ctx, job, src, False, job.values["radius"] / math.sqrt(3.0) * scale(job))
        _, _, mean = reduce(ctx, job, src, source_size(job, "input"))
        ctx.dispatch(FINAL, job, job.values, srcs={"blurred": (blurred, False), "mean": (mean, False)})


FINAL = helper("luminance_highpass_final", "luminance_highpass.wgsl",
               {"input": Input(PortType.COLOR), "blurred": Input(PortType.COLOR), "mean": Input(PortType.COLOR)},
               {"output": Output(PortType.COLOR)})
