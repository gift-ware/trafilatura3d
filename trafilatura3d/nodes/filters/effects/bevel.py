from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import BLUR_HQ_RADIUS, Effect, gauss, helper, nearest_seeds, scale, source


class Bevel(Effect):
    """Bevels the edges of a (black and white) height map and gives the result and its normal map:
    a linear ramp over 128 x |Distance| pixels at 256 from the shapes' edge (jump
    flooding on the input cut at 0.5): Distance > 0 grows it outwards (1 on the shapes, falling to 0),
    < 0 bevels into them (0 off them, rising to 1). Corner Type: Round = Euclidean distance (rounded
    corners), Angular = the larger axis offset (square corners). Use Custom Curve maps the 0-1 ramp
    through the Custom Curve input (read along its width, as from a Gradient Linear). Smoothing blurs
    the result like Blur HQ of that Intensity; Use Non-Uniform Blur keeps it off where the ramp is 0.
    Normal: Sobel of the bevelled height x Normal Intensity, in Normal Format."""

    type = "bevel"
    label = "Bevel (Filter Node)"
    category = "Filters/Effects"
    catalog = "bevel"
    inputs = {"input": Input(PortType.GRAY, "input"), "custom_curve": Input(PortType.GRAY, "Custom Curve")}
    outputs = {"height": Output(PortType.GRAY, "Height"), "normal": Output(PortType.COLOR, "Normal")}
    params = {
        "distance": Param(ParamKind.FLOAT, 0.5, "Distance", -1.0, 1.0),
        "corner_type": Param(ParamKind.ENUM, "Round", "Corner Type", choices=("Round", "Angular")),
        "smoothing": Param(ParamKind.FLOAT, 0.0, "Smoothing", 0.0, 5.0),
        "use_non_uniform_blur": Param(ParamKind.BOOL, True, "Use Non-Uniform Blur"),
        "use_custom_curve": Param(ParamKind.BOOL, False, "Use Custom Curve"),
        "normal_intensity": Param(ParamKind.FLOAT, 10.0, "Normal Intensity", 0.0, 50.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
    }

    def encode(self, ctx, job):
        v = job.values
        seeds = nearest_seeds(ctx, job, source(ctx, job, "input"))
        h = ctx.scratch(job, True)
        ctx.dispatch(PROFILE, job, v, srcs={"seeds": (seeds, False)}, dsts={"out": h})
        blurred = h
        if v["smoothing"] > 0.0:
            blurred = gauss(ctx, job, (h, True), BLUR_HQ_RADIUS * v["smoothing"] * scale(job))
        ctx.dispatch(FINAL, job, v, srcs={"raw": (h, True), "blurred": (blurred, True)})


PROFILE = helper("fx_bevel_profile", "_bevel_profile.wgsl",
                 {"seeds": Input(PortType.COLOR), "custom_curve": Input(PortType.GRAY)},
                 {"out": Output(PortType.GRAY)}, params=Bevel.params)


class _Final(Effect):
    type = "fx_bevel_final"
    registered = False
    inputs = {"raw": Input(PortType.GRAY), "blurred": Input(PortType.GRAY)}
    outputs = Bevel.outputs
    params = Bevel.params


FINAL = _Final()
