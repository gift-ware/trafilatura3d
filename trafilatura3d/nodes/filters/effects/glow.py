from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import BLUR_HQ_RADIUS, Effect, gauss, helper, scale


class Glow(Effect):
    """Outer glow around a black and white mask: the input's luminance cut at Clear
    Amount (a 0.02 wide ramp), blurred like Blur HQ of Glow Size, added to the input at Glow Amount: in
    Glow Color for color (the input still a black and white mask), as gray for
    grayscale."""

    type = "glow"
    label = "Glow"
    category = "Filters/Effects"
    catalog = "glow"
    inputs = {"input": Input(label="Input")}
    params = {
        "glow_amount": Param(ParamKind.FLOAT, 0.5, "Glow Amount", 0.0, 1.0),
        "clear_amount": Param(ParamKind.FLOAT, 0.5, "Clear Amount", 0.0, 1.0),
        "glow_size": Param(ParamKind.FLOAT, 10.0, "Glow Size", 0.0, 20.0),
        "glow_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Glow Color"),
    }

    def encode(self, ctx, job):
        mask = ctx.scratch(job, True)
        ctx.dispatch(MASK, job, job.values, dsts={"out": mask})
        blur = gauss(ctx, job, (mask, True), BLUR_HQ_RADIUS * job.values["glow_size"] * scale(job))
        ctx.dispatch(FINAL, job, job.values, srcs={"blur": (blur, True)})


MASK = helper("fx_glow_mask", "_glow_mask.wgsl", {"input": Input()}, {"out": Output(PortType.GRAY)},
              params=Glow.params)


class _Final(Effect):
    type = "fx_glow_final"
    registered = False
    inputs = {"input": Input(), "blur": Input(PortType.GRAY)}
    params = Glow.params


FINAL = _Final()
