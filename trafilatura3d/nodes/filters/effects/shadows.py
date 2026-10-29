import math

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect, scale

# Fitted on the output, on binary shapes and continuous heightmaps over the parameter ranges:
REACH = 0.135  # how far the shadow reaches: pixels at 256 per unit of 100 x Shadow Distance, at 2 Samples
SPREAD = 0.028  # each Sample past 2 lengthens the reach by this share
SAMPLES = 3.1  # the shadow darkens with Samples as 1 - exp(-Samples / SAMPLES)
GAIN = 0.96  # the shadow's strength at 8 Samples and a single direction
CONE_GAIN = 0.38  # the strength grows by this share as Edges Softness opens the cone
PEAK = 0.06  # where the shadow sits between the cone's mean and its darkest ray, for a single direction ...
PEAK_CONE = 0.13  # ... and how much nearer the darkest ray an open cone takes it


class Shadows(Effect):
    """The shadow the input's heights cast, in one pass of ours: along 7 rays of a cone opening from the pixel
    towards the light (Light Angle, turns), how much the heights stand above the pixel's, Samples steps up to
    REACH x 100 x Shadow Distance pixels at 256 (a little farther with more Samples), the far steps weighing less;
    between the cone's mean and its darkest ray. Edges Softness closes the cone (0: all around, 1: a single
    direction); fewer Samples give a lighter shadow. Slopes and smooth bumps cast none, steps do. White, with the
    shadow in black."""

    type = "shadows"
    label = "Shadows (Filter Node)"
    category = "Filters/Effects"
    catalog = "shadows"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "shadow_distance": Param(ParamKind.FLOAT, 0.3, "Shadow Distance", 0.0, 1.0),
        "light_angle": Param(ParamKind.FLOAT, 0.13, "Light Angle", 0.0, 1.0),
        "edges_softness": Param(ParamKind.FLOAT, 1.0, "Edges Softness", 0.0, 1.0),
        "samples": Param(ParamKind.INT, 8, "Samples", 1, 16),
    }
    internal_uniforms = {"theta": ParamKind.FLOAT, "cone": ParamKind.FLOAT, "reach": ParamKind.FLOAT,
                         "gain": ParamKind.FLOAT, "peak": ParamKind.FLOAT}

    def encode(self, ctx, job):
        v = job.values
        opening = 1.0 - v["edges_softness"]
        n = min(max(v["samples"], 1), 16)
        reach_px = REACH * 100.0 * v["shadow_distance"] * scale(job) * (1.0 + SPREAD * (n - 2))
        gain = GAIN * (1.0 + CONE_GAIN * opening) * (1.0 - math.exp(-n / SAMPLES)) / (1.0 - math.exp(-8.0 / SAMPLES))
        ctx.dispatch(self, job, {**v, "theta": (v["light_angle"] + 0.5) * 2.0 * math.pi, "cone": math.pi * opening,
                                 "reach": reach_px / job.size[0], "gain": gain,
                                 "peak": min(max(PEAK + PEAK_CONE * opening, 0.0), 1.0)})
