from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import downsample

from ._fx import Effect, source

MIPS = 11  # samples up to 2^11 px away


class AmbientOcclusionHBAO(Effect):
    """Horizon-based ambient occlusion of a height map: in 4, 8 or 16 evenly spaced directions
    (Quality), marching out to points 2, 4, 8 ... px away, each read on the height box-averaged over as many
    pixels (a mip, so far points see broad forms); every rise of the horizon adds its gain in sine, weighted by a
    distance falloff (full up to 0.4 x Radius x size, linear to 0 at Radius x size; Radius 1 = the image's
    width). AO = 1 - the mean over directions: slopes going down count negative, so a tilted plane stays
    white. Height: Height
    Depth x the image width, or with Use World Units Height Scale (cm) over Surface Size (cm). GPU
    Optimization changes nothing here.
    Non-Square (non-square images): on, the points are the same number of pixels away along both axes;
    off, the same UV distance (2^k over the larger side, with no extra multiplier)."""

    type = "ambient_occlusion_hbao"
    label = "Ambient Occlusion (HBAO) (Filter Node)"
    category = "Filters/Effects"
    catalog = "ambient_occlusion_hbao"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "use_world_units": Param(ParamKind.BOOL, False, "Use World Units"),
        "height_depth": Param(ParamKind.FLOAT, 0.1, "Height Depth", 0.0, 1.0),
        "surface_size": Param(ParamKind.FLOAT, 300.0, "Surface Size", 0.0, 1000.0),
        "height_scale_cm": Param(ParamKind.FLOAT, 30.0, "Height Scale (cm)", 0.0, 1000.0),
        "radius": Param(ParamKind.FLOAT, 1.0, "Radius", 0.0, 1.0),
        "quality": Param(ParamKind.ENUM, "8 samples", "Quality", choices=("4 samples", "8 samples", "16 samples")),
        "gpu_optimization": Param(ParamKind.BOOL, False, "GPU Optimization"),
        "non_square": Param(ParamKind.BOOL, True, "Non-Square"),
    }

    def encode(self, ctx, job):
        v = job.values
        side = min(job.size)
        # Points 2^k px away while 2^k < Radius x side (beyond, the falloff is 0).
        levels = 0
        while levels < MIPS and 2 ** (levels + 1) < v["radius"] * side:
            levels += 1
        mips, cur = {}, source(ctx, job, "input")
        for k in range(1, MIPS + 1):
            if k <= levels:
                cur = (downsample(ctx, job, cur[0], True, (2, 2)), True)
            mips[f"m{k}"] = cur  # unused levels repeat the last one (never read)
        ctx.dispatch(FINAL, job, {**v, "levels": levels}, srcs=mips)


class _Final(Effect):
    type = "fx_ambient_occlusion_hbao_final"
    registered = False
    inputs = {"input": Input(PortType.GRAY), **{f"m{k}": Input(PortType.GRAY) for k in range(1, MIPS + 1)}}
    outputs = AmbientOcclusionHBAO.outputs
    params = AmbientOcclusionHBAO.params
    internal_uniforms = {"levels": ParamKind.INT}


FINAL = _Final()
