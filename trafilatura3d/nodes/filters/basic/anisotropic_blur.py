import math

from trafilatura3d.nodes.atomic.directional_blur import directional_blur
from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import GAUSS, BasicNode, blur2d, scale, source

SIGMA = 0.57  # Quality 1: the gaussian's sigma per unit of Intensity (fitted on white noise, 1024 px)


class AnisotropicBlur(BasicNode):
    """A Directional Blur of Intensity (pixels at 256 px on each side) along Angle (turns), then one of
    Intensity x (1 - Anisotropy) across it, so Anisotropy 0 is a regular (box) blur. Quality 1: a smoother
    kernel, a gaussian of sigma SIGMA x those intensities along and across Angle (fitted). Intensity 0: the
    input."""

    type = "anisotropic_blur"
    label = "Anisotropic Blur"
    category = "Filters/Blurs"
    catalog = "anisotropic_blur"
    inputs = {"input": Input(label="Input")}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 16.0),
        "anisotropy": Param(ParamKind.FLOAT, 0.5, "Anisotropy", 0.0, 1.0),
        "angle": Param(ParamKind.FLOAT, 0.0, "Angle", 0.0, 1.0),
        "quality": Param(ParamKind.INT, 0, "Quality", 0, 1),
    }

    def encode(self, ctx, job):
        v = job.values
        gray = job.is_gray("output")
        along = max(v["intensity"], 0.0)
        across = along * (1.0 - v["anisotropy"])
        out = ctx.ev.texture(job, "output")
        src = source(ctx, job, "input")
        if v["quality"] >= 1:  # a gaussian along Angle and across it (the one blur of every node)
            a = 2.0 * math.pi * v["angle"]
            k = SIGMA * scale(job)
            blur2d(ctx, job, src, gray, k * along, GAUSS, dst=out, u=(math.cos(a), math.sin(a)), radius_v=k * across)
            return
        cur = src
        for i, (intensity, angle) in enumerate(((along, v["angle"]), (across, v["angle"] + 0.25))):
            target = out if i == 1 else ctx.scratch(job, gray)
            directional_blur(ctx, job, cur, gray, intensity, angle, dst=target)
            cur = (target, gray)
