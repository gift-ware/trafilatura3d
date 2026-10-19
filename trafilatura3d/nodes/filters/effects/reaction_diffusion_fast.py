from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect, gauss, helper, scale

ITERATIONS = 16


class ReactionDiffusionFast(Effect):
    """Fast reaction-diffusion: the input, contrasted by Contrast, is iterated 16 times through an
    activator-inhibitor step (the difference of a gaussian blur at Radius and one at twice Radius,
    amplified and clamped), which grows spots and stripes about Radius wide (pixels at 256) out of the
    input's structure, like animal skin patterns. For speed the iterations run on a reduced image where
    Radius is about 3 pixels, then the result is upsampled and sharpened back."""

    type = "reaction_diffusion_fast"
    label = "Reaction Diffusion Fast"
    category = "Filters/Effects"
    catalog = "reaction_diffusion_fast"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY, "Output")}
    params = {
        "radius": Param(ParamKind.FLOAT, 4.0, "Radius", 1.0, 32.0),
        "contrast": Param(ParamKind.FLOAT, 0.5, "Contrast", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        v = job.values
        w, h = job.size
        r_px = max(v["radius"], 0.1) * scale(job)
        f = max(1.0, r_px / 3.0)
        size = (max(16, min(w, int(round(w / f)))), max(16, min(h, int(round(h / f)))))
        r = r_px * size[0] / w
        a, b = ctx.scratch(job, True, size=size), ctx.scratch(job, True, size=size)
        ctx.dispatch(STEP, job, {**v, "init": 1}, dsts={"out": a}, size=size)
        for _ in range(ITERATIONS):
            small = gauss(ctx, job, (a, True), r, size=size)
            large = gauss(ctx, job, (a, True), 2.0 * r, size=size)
            ctx.dispatch(STEP, job, {**v, "init": 0}, srcs={"small": (small, True), "large": (large, True)},
                         dsts={"out": b}, size=size)
            a, b = b, a
        ctx.dispatch(FINAL, job, v, srcs={"state": (a, True)})


STEP = helper("fx_rd_step", "_rd_step.wgsl",
              {"input": Input(PortType.GRAY), "small": Input(PortType.GRAY), "large": Input(PortType.GRAY)},
              {"out": Output(PortType.GRAY)}, {"init": ParamKind.INT}, params=ReactionDiffusionFast.params)


class _Final(Effect):
    type = "fx_reaction_diffusion_fast_final"
    registered = False
    inputs = {"state": Input(PortType.GRAY)}
    outputs = ReactionDiffusionFast.outputs
    params = ReactionDiffusionFast.params


FINAL = _Final()
