import math

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect, scale

REACH = 0.135  # how far the shadow reaches: pixels at 256 per unit of 100 x Shadow Distance (fitted on the output)
SAMPLES = 3.8  # the shadow darkens with Samples as 1 - exp(-Samples / SAMPLES) (fitted, 1 at the default 8)


class Shadows(Effect):
    """The shadow a binary shape casts, alone, in one pass of ours: off the shape, the share of samples that fall
    on it along 7 rays of a cone opening from the pixel towards the light (Light Angle, turns), halfway between
    the cone's mean and its darkest ray, Samples steps up to REACH x 100 x Shadow Distance pixels at 256; Edges
    Softness closes the cone (0: all around, 1: a single direction); fewer Samples give a lighter shadow. White,
    with the shadow in black."""

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
                         "gain": ParamKind.FLOAT}

    def encode(self, ctx, job):
        v = job.values
        opening = 1.0 - v["edges_softness"]
        reach_px = REACH * 100.0 * v["shadow_distance"] * scale(job)
        gain = (1.0 - math.exp(-v["samples"] / SAMPLES)) / (1.0 - math.exp(-8.0 / SAMPLES))
        ctx.dispatch(self, job, {**v, "theta": (v["light_angle"] + 0.5) * 2.0 * math.pi, "cone": math.pi * opening,
                                 "reach": reach_px / job.size[0], "gain": gain})
