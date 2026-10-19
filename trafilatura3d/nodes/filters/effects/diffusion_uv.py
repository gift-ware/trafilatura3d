from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._diffusion import MODE_UV, diffuse
from ._fx import Effect


class DiffusionUV(Effect):
    """Interpolates UV coordinates (red = U, green = V) of the Source from the pixels where the Mask is\n    white (cut at 0.5) into the black ones, independently of pixel positions. On tiled axes coordinates are\n    averaged around the 0/1 seam (0.1 and 0.8 give 0.95), so known points should be less than half the image\n    apart. Output: U, V, 0, 1.
    See _diffusion.py: Iterations passes from Distance (fraction of the image) down to one pixel,
    Intensity scaling the reach locally, Enable Dithering for fewer passes with some noise."""

    type = "diffusion_uv"
    label = "Diffusion UV"
    category = "Filters/Effects"
    catalog = "diffusion_uv"
    libs = ("_diff.wgsl",)
    inputs = {"source": Input(PortType.COLOR, "Source"), "mask": Input(PortType.GRAY, "Mask")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "iterations": Param(ParamKind.INT, 16, "Iterations", 0, 64),

    }

    def encode(self, ctx, job):
        v = job.values
        state = diffuse(ctx, job, MODE_UV, v["iterations"], 1.0, False)
        ctx.dispatch(FINAL, job, {**v, "mode": MODE_UV, "alpha_mask": int(v.get("use_alpha_as_mask", False))},
                     srcs={"v": (state, False)})


class _Final(Effect):
    type = "fx_diffusion_uv_final"
    registered = False
    libs = ("_diff.wgsl",)
    inputs = {"source": Input(PortType.COLOR), "mask": Input(PortType.GRAY, default=(1.0, 1.0, 1.0, 1.0)),
              "v": Input(PortType.COLOR)}
    outputs = DiffusionUV.outputs
    internal_uniforms = {"mode": ParamKind.INT, "alpha_mask": ParamKind.INT}


FINAL = _Final()
