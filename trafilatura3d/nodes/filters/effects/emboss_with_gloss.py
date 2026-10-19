from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ..basic._lib import blur2d
from ._fx import Effect, helper, scale

SMOOTH = 0.5  # the light is smoothed by a gaussian of this sigma (pixels at 256) before the bands (fitted)


class EmbossWithGloss(Effect):
    """Fake lighting baked into the Color input from the Height input, ours. The light is the height's slope
    towards Light Angle times Intensity: where it faces the light, a sheen in Highlight Color; where it faces
    away, a shade towards Shadow Color. Gloss turns the sheen into chrome-like bands: a stripe pattern read
    through the direction and value of the smoothed light (as a reflection reads its surroundings), so the bands
    follow the relief's edges and fold where it bends, denser and finer with Gloss, solid where the light is
    strong; the plain sheen fades as Gloss grows. Constants fitted on the output. Intensity 0: the color as is."""

    type = "emboss_with_gloss"
    label = "Emboss With Gloss"
    category = "Filters/Effects"
    catalog = "emboss_with_gloss"
    inputs = {"color": Input(PortType.COLOR, "Color", default=(0.5, 0.5, 0.5, 1.0)),
              "height": Input(PortType.GRAY, "Height")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "highlight_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Highlight Color"),
        "shadow_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Shadow Color"),
        "gloss": Param(ParamKind.FLOAT, 0.25, "Gloss", 0.0, 0.5),
        "intensity": Param(ParamKind.FLOAT, 5.0, "Intensity", 0.0, 10.0),
        "light_angle": Param(ParamKind.FLOAT, 0.0, "Light Angle", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        v = job.values
        h = job.inputs.get("height")
        height = (ctx.ev.texture(h[0], h[1]), True) if h else (ctx.ev.constant((0.0, 0.0, 0.0, 1.0)), True)
        lit = ctx.scratch(job, True)
        ctx.dispatch(LIGHT, job, {"intensity": v["intensity"], "light_angle": v["light_angle"]},
                     srcs={"height": height}, dsts={"out": lit})
        smooth = blur2d(ctx, job, (lit, True), True, SMOOTH * scale(job))
        ctx.dispatch(FINAL, job, v, srcs={"height": height, "smooth": (smooth, True)})


LIGHT = helper("fx_ewg_light", "_ewg_light.wgsl", {"height": Input(PortType.GRAY)}, {"out": Output(PortType.GRAY)},
               {"intensity": ParamKind.FLOAT, "light_angle": ParamKind.FLOAT}, libs=("_ewg_slope.wgsl",))


class _Final(Effect):
    type = "fx_emboss_with_gloss_final"
    registered = False
    libs = ("_ewg_slope.wgsl",)
    inputs = {"color": EmbossWithGloss.inputs["color"], "height": Input(PortType.GRAY),
              "smooth": Input(PortType.GRAY)}
    outputs = EmbossWithGloss.outputs
    params = EmbossWithGloss.params


FINAL = _Final()
