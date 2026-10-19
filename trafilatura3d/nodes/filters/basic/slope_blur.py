from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode, helper

SPREAD = 0.7  # the samples weigh a gaussian of the distance along the path, sigma SPREAD x its length (fitted);
# the same in every mode, so that Min <= Blur <= Max everywhere


class SlopeBlur(BasicNode):
    """Blur that follows the Slope map like a heightmap: from the pixel, a path of Samples steps along the slope's
    gradient (each as long as a Warp of Intensity / Samples moves: shorter where the slope is gentle), the values
    met combined: Blur a mean weighted by a gaussian of the distance along the path, Min / Max the same mean of the
    extreme met so far (an erosion / dilation growing along the path: dark areas eat away / white areas smear
    out). Defaults 8, 10 (the documentation gives none). No slope or Intensity 0: unchanged."""

    type = "slope_blur"
    label = "Slope Blur"
    category = "Filters/Blurs"
    catalog = "slope_blur"
    inputs = {"input": Input(label="Input"), "slope": Input(PortType.GRAY, "Slope")}
    params = {
        "samples": Param(ParamKind.INT, 8, "Samples", 0, 32),
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 16.0),
        "mode": Param(ParamKind.ENUM, "Blur", "Mode", choices=("Blur", "Min", "Max")),
    }
    def encode(self, ctx, job):
        """The slope's gradient once (`GRADIENT`, at the Slope map's size), then the walk (`WALK`) reading it."""
        src = job.inputs.get("slope")
        size = src[0].size if src else (1, 1)
        # half floats: the gradient reaches about half the Slope map's width (2048 at 4096), a thousandth relative
        gradient = ctx.scratch(job, True, "16f", size, alpha=True)
        ctx.dispatch(GRADIENT, job, {}, dsts={"output": gradient}, size=size)
        ctx.dispatch(WALK, job, {**job.values, "spread": SPREAD}, srcs={"gradient": (gradient, True)})


GRADIENT = helper("slope_blur_gradient", "slope_blur_gradient.wgsl", {"slope": SlopeBlur.inputs["slope"]},
                  {"output": Output(PortType.GRAY_ALPHA, "Gradient")})
WALK = helper("slope_blur_walk", "slope_blur.wgsl",
              {"input": SlopeBlur.inputs["input"], "gradient": Input(PortType.GRAY_ALPHA, "Gradient")},
              {"output": Output(label="Output")}, SlopeBlur.params, {"spread": ParamKind.FLOAT})
