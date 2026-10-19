from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode, blur2d, helper, reduce, scale, source, source_size

SIGMA = 0.58  # the local mean: a gaussian of sigma SIGMA x Cancellation Radius, pixels at 256 (fitted on the output)
GAIN = 2.5  # the detail is applied as an exposure, mean x exp(GAIN x detail) (fitted)


class LightingCancelLowFrequencies(BasicNode):
    """Highpass that keeps the colours: per channel, the local mean (a gaussian of sigma SIGMA x Cancellation
    Radius, pixels at 256) is replaced by the image's mean colour and the detail around it applied as an exposure
    (mean x exp(GAIN x detail): lighting is multiplicative, so darker detail is compressed and brighter detail
    opened up), so large lighting gradients go and the overall tint stays."""

    type = "lighting_cancel_low_frequencies"
    label = "Lighting Cancel Low Frequencies"
    category = "Filters/Adjustments"
    catalog = "lighting_cancel_low_frequencies"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"cancellation_radius": Param(ParamKind.FLOAT, 16.0, "Cancellation Radius", 0.0, 64.0)}

    def encode(self, ctx, job):
        src = source(ctx, job, "input")
        sigma = job.values["cancellation_radius"] * SIGMA * scale(job)
        blurred = blur2d(ctx, job, src, False, sigma, taps=16)  # a wide smooth blur needs few taps
        _, _, mean = reduce(ctx, job, src, source_size(job, "input"))
        ctx.dispatch(FINAL, job, {**job.values, "gain": GAIN}, srcs={"blurred": (blurred, False),
                                                                  "mean": (mean, False)})


FINAL = helper("lighting_cancel_low_final", "lighting_cancel_low_frequencies.wgsl",
               {"input": Input(PortType.COLOR), "blurred": Input(PortType.COLOR), "mean": Input(PortType.COLOR)},
               {"output": Output(PortType.COLOR)}, internal={"gain": ParamKind.FLOAT})
