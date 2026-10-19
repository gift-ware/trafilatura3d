from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect, helper
from .flood_fill import F32


class Texture3DSDF(Effect):
    """Distance field of a shape stored as a 3D texture (an atlas of Grid Size slices, see 3D Texture
    Position): the Mask Input above Threshold is inside. The nearest surface voxel of every voxel is
    found by 3D jump flooding on the GPU (26 neighbours at halving steps), and the distance is in
    unit-cube units. Distance Field: distance outside, 0 inside; Signed Distance Field: positive
    outside, negative inside. 32F by default (negative values)."""

    type = "3d_texture_sdf"
    label = "3D Texture SDF"
    category = "Filters/Effects"
    catalog = "3d_texture_sdf"
    libs = ("_volume.wgsl", "_volume_grid.wgsl")
    inputs = {"mask_input": Input(PortType.GRAY, "Mask Input")}
    outputs = {"output": Output(PortType.GRAY)}
    base_defaults = F32
    params = {
        "threshold": Param(ParamKind.FLOAT, 0.5, "Threshold", 0.0, 1.0),
        "grid_size": Param(ParamKind.INT2, [16, 16], "Grid Size", 1, 16),
        "output": Param(ParamKind.ENUM, "Signed Distance Field", "Output",
                        choices=("Distance Field", "Signed Distance Field")),
    }

    def encode(self, ctx, job):
        v = job.values
        w, h = job.size
        gx, gy = (max(int(g), 1) for g in v["grid_size"])
        a, b = ctx.scratch(job, False), ctx.scratch(job, False)
        ctx.dispatch(SEED, job, v, dsts={"seeds": a})
        step = max(w // gx, h // gy, gx * gy) // 2
        while step >= 1:
            ctx.dispatch(FLOOD, job, {**v, "step": step}, srcs={"seeds": (a, False)}, dsts={"seeds": b})
            a, b, step = b, a, step // 2
        ctx.dispatch(FLOOD, job, {**v, "step": 1}, srcs={"seeds": (a, False)}, dsts={"seeds": b})
        ctx.dispatch(FINAL, job, v, srcs={"seeds": (b, False)})


SEED = helper("fx_sdf3_seed", "_sdf3_seed.wgsl", {"mask_input": Input(PortType.GRAY)},
              {"seeds": Output(PortType.COLOR)}, params=Texture3DSDF.params, libs=Texture3DSDF.libs)
FLOOD = helper("fx_sdf3_flood", "_sdf3_flood.wgsl", {"seeds": Input(PortType.COLOR)},
               {"seeds": Output(PortType.COLOR)}, {"step": ParamKind.INT}, params=Texture3DSDF.params,
               libs=Texture3DSDF.libs)


class _Final(Effect):
    type = "fx_3d_texture_sdf_final"
    registered = False
    libs = Texture3DSDF.libs
    inputs = {"mask_input": Input(PortType.GRAY), "seeds": Input(PortType.COLOR)}
    outputs = Texture3DSDF.outputs
    params = Texture3DSDF.params


FINAL = _Final()
