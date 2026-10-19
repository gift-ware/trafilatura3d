import math

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode, blur2d, helper, scale, source


class LightingCancelHighFrequencies(BasicNode):
    """Cancel small bright lighting details of a colour image: every channel that is above its surroundings
    (a gaussian of sigma Radius / sqrt 3, pixels at 256 px) is pulled towards them by Intensity (1: down to the
    surroundings); what is darker than its surroundings stays."""

    type = "lighting_cancel_high_frequencies"
    label = "Lighting Cancel High Frequencies"
    category = "Filters/Adjustments"
    catalog = "lighting_cancel_high_frequencies"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "intensity": Param(ParamKind.FLOAT, 1.0, "Intensity", 0.0, 1.0),
        "radius": Param(ParamKind.FLOAT, 7.0, "Radius", 0.0, 10.0),
    }

    def encode(self, ctx, job):
        src = source(ctx, job, "input")
        blurred = blur2d(ctx, job, src, False, job.values["radius"] / math.sqrt(3.0) * scale(job))
        ctx.dispatch(FINAL, job, job.values, srcs={"blurred": (blurred, False)})


FINAL = helper("lighting_cancel_high_final", "lighting_cancel_high_frequencies.wgsl",
               {"input": Input(PortType.COLOR), "blurred": Input(PortType.COLOR)}, {"output": Output(PortType.COLOR)},
               params=LightingCancelHighFrequencies.params)
