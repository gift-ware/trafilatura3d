import math

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import downsample

from ._fx import Effect, source

MIPS = 11  # down to the 2 px level of a 4096 image
GAIN = 0.135  # weight of every octave's divergence (fitted, see CurvatureSmooth)


class CurvatureSmooth(Effect):
    """Smooth curvature of a normal map, our method: on every mip level of the normal (box-averaged down to
    2 px), the divergence of its XY from four bilinear samples on the diagonals half a texel away; the octaves
    are summed with weight GAIN (fitted on noise, cells and bricks), so broad
    forms and fine details both show. Radius is how many octaves count, as a
    share of log2(size) (1 = all, 0 = only the finest). Curvature: flat 0.5, convex brighter, concave darker;
    Convexity / Concavity: the two halves stretched to 0-1."""

    type = "curvature_smooth"
    label = "Curvature Smooth"
    category = "Filters/Effects"
    catalog = "curvature_smooth"
    inputs = {"normal": Input(PortType.COLOR, "Normal", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"curvature": Output(PortType.GRAY, "Curvature"), "convexity": Output(PortType.GRAY, "Convexity"),
               "concavity": Output(PortType.GRAY, "Concavity")}
    params = {
        "radius": Param(ParamKind.FLOAT, 1.0, "Radius", 0.0, 1.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal format", choices=("DirectX", "OpenGL")),
    }

    def encode(self, ctx, job):
        side = min(job.size)
        count = max(side.bit_length() - 1, 1)  # log2(size) levels, from the image to 2 px
        mips, cur = {}, source(ctx, job, "normal")
        for k in range(1, MIPS + 1):
            if k < count:
                cur = (downsample(ctx, job, cur[0], cur[1], (2, 2)), cur[1])
            mips[f"m{k}"] = cur  # unused levels repeat the last one (never read)
        ctx.dispatch(FINAL, job, {**job.values, "log2_size": math.log2(side), "levels": min(count, MIPS + 1),
                                  "gain": GAIN},
                     srcs=mips)


class _Final(Effect):
    type = "fx_curvature_smooth_final"
    registered = False
    inputs = {"normal": Input(PortType.COLOR), **{f"m{k}": Input(PortType.COLOR) for k in range(1, MIPS + 1)}}
    outputs = CurvatureSmooth.outputs
    params = CurvatureSmooth.params
    internal_uniforms = {"log2_size": ParamKind.FLOAT, "levels": ParamKind.INT, "gain": ParamKind.FLOAT}


FINAL = _Final()
