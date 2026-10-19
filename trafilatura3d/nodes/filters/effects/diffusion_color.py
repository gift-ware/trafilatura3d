from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._diffusion import MODE_COLOR, diffuse
from ._fx import Effect


class DiffusionColor(Effect):
    """Smooth gradations of the Source's colors from the pixels where the Mask is white (cut at 0.5; or\n    where the Source's alpha is, with Use Alpha as Mask) into the black ones. Is Normal Map renormalizes the\n    vectors every pass.
    See _diffusion.py: Iterations passes from Distance (fraction of the image) down to one pixel,
    Intensity scaling the reach locally, Enable Dithering for fewer passes with some noise."""

    type = "diffusion_color"
    label = "Diffusion Color"
    category = "Filters/Effects"
    catalog = "diffusion_color"
    libs = ("_diff.wgsl",)
    inputs = {"source": Input(PortType.COLOR, "Source"), "mask": Input(PortType.GRAY, "Mask"),
              "intensity": Input(PortType.GRAY, "Intensity", default=(1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "iterations": Param(ParamKind.INT, 16, "Iterations", 0, 64),
        "distance": Param(ParamKind.FLOAT, 1.0, "Distance", 0.0, 1.0),
        "enable_dithering": Param(ParamKind.BOOL, True, "Enable Dithering"),
        "is_normal_map": Param(ParamKind.BOOL, False, "Is Normal Map"),
        "use_alpha_as_mask": Param(ParamKind.BOOL, False, "Use Alpha as Mask"),
    }

    def encode(self, ctx, job):
        v = job.values
        state = diffuse(ctx, job, MODE_COLOR, v["iterations"], v["distance"], v["enable_dithering"],
                    v["is_normal_map"], v["use_alpha_as_mask"])
        ctx.dispatch(FINAL, job, {**v, "mode": MODE_COLOR, "alpha_mask": int(v.get("use_alpha_as_mask", False))},
                     srcs={"v": (state, False)})


class _Final(Effect):
    type = "fx_diffusion_color_final"
    registered = False
    libs = ("_diff.wgsl",)
    inputs = {"source": Input(PortType.COLOR), "mask": Input(PortType.GRAY, default=(1.0, 1.0, 1.0, 1.0)),
              "v": Input(PortType.COLOR)}
    outputs = DiffusionColor.outputs
    internal_uniforms = {"mode": ParamKind.INT, "alpha_mask": ParamKind.INT}


FINAL = _Final()
