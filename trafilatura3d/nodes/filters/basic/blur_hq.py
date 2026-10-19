import math

from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import GAUSS, BasicNode, blur2d, scale, source

# Our method: one isotropic gaussian, its width fitted on white noise at 256-2048 px: about 0.6 x Intensity
# pixels at 256, plus a softening of about PIXEL output pixels (added in quadrature) that the node also has, reached by Intensity 1 / (PIXEL_RAMP x scale) (Intensity 0: none).
SIGMA = 0.6
PIXEL = 1.1
PIXEL_RAMP = 1.6


class BlurHQ(BasicNode):
    """High quality blur: an isotropic gaussian (the one blur of every node, `_lib.blur2d`) of sigma 0.6 x
    Intensity (pixels at 256 px) plus about a pixel of softening (fitted). Quality is kept for the files: our
    gaussian is already round, both values give it. Intensity default 10. Intensity 0: the input."""

    type = "blur_hq"
    label = "Blur HQ"
    category = "Filters/Blurs"
    catalog = "blur_hq"
    inputs = {"input": Input(label="Input")}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 16.0),
        "quality": Param(ParamKind.INT, 0, "Quality", 0, 1),
    }

    def encode(self, ctx, job):
        blur_hq(ctx, job, source(ctx, job, "input"), job.is_gray("output"), job.values["intensity"],
                job.values["quality"], ctx.ev.texture(job, "output"))


def blur_hq(ctx, job, src, gray: bool, intensity: float, quality: int, out):
    """Blur HQ of `src` ((texture, gray)) at the job's size into `out` (also for the nodes that use it inside:
    Height Map Frequencies Mapper, Color Match). `quality` is accepted and has no effect (see BlurHQ)."""
    r = max(intensity, 0.0) * scale(job)
    sigma = math.hypot(SIGMA * r, PIXEL * min(1.0, PIXEL_RAMP * r))
    return blur2d(ctx, job, src, gray, sigma, GAUSS, dst=out)
