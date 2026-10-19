from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._diffusion import MODE_GRAY, diffuse
from ._fx import Effect


class DiffusionGrayscale(Effect):
    """Smooth gradations of the Source's values from the pixels where the Mask is white (cut at 0.5) into\n    the black ones.
    See _diffusion.py: Iterations passes from Distance (fraction of the image) down to one pixel,
    Intensity scaling the reach locally, Enable Dithering for fewer passes with some noise."""

    type = "diffusion_grayscale"
    label = "Diffusion Grayscale"
    category = "Filters/Effects"
    catalog = "diffusion_grayscale"
    libs = ("_diff.wgsl",)
    inputs = {"source": Input(PortType.GRAY, "Source"), "mask": Input(PortType.GRAY, "Mask"),
              "intensity": Input(PortType.GRAY, "Intensity", default=(1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "iterations": Param(ParamKind.INT, 16, "Iterations", 0, 64),
        "distance": Param(ParamKind.FLOAT, 1.0, "Distance", 0.0, 1.0),
        "enable_dithering": Param(ParamKind.BOOL, True, "Enable Dithering"),

    }

    def encode(self, ctx, job):
        v = job.values
        state = diffuse(ctx, job, MODE_GRAY, v["iterations"], v["distance"], v["enable_dithering"])
        ctx.dispatch(FINAL, job, {**v, "mode": MODE_GRAY, "alpha_mask": int(v.get("use_alpha_as_mask", False))},
                     srcs={"v": (state, False)})


class _Final(Effect):
    type = "fx_diffusion_grayscale_final"
    registered = False
    libs = ("_diff.wgsl",)
    inputs = {"source": Input(PortType.COLOR), "mask": Input(PortType.GRAY, default=(1.0, 1.0, 1.0, 1.0)),
              "v": Input(PortType.COLOR)}
    outputs = DiffusionGrayscale.outputs
    internal_uniforms = {"mode": ParamKind.INT, "alpha_mask": ParamKind.INT}


FINAL = _Final()
