from trafilatura3d.nodes.base import Input, Output, Param, ParamKind

from ._lib import BOX, BasicNode, blur2d, helper, scale, source


class Highpass(BasicNode):
    """Photoshop-like highpass: input - blur(input) + 0.5 (an Add Sub of the inverted
    blur at half opacity), the blur being the atomic Blur: a box of half-width Radius in pixels of a
    256 px image. Alpha passes through."""

    type = "highpass"
    label = "Highpass"
    category = "Filters/Adjustments"
    catalog = "highpass"
    inputs = {"input": Input(label="Input")}
    params = {"radius": Param(ParamKind.FLOAT, 6.0, "Radius", 0.0, 64.0)}

    def encode(self, ctx, job):
        src = source(ctx, job, "input")
        blurred = blur2d(ctx, job, src, src[1], job.values["radius"] * scale(job), BOX)
        ctx.dispatch(FINAL, job, job.values, srcs={"blurred": (blurred, src[1])})


FINAL = helper("highpass_final", "highpass.wgsl", {"input": Input(), "blurred": Input()}, {"output": Output()})
