from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.filters.effects.fxaa import FXAA
from trafilatura3d.nodes.paint._lib import DOC, B, Filter

from ._passes import helper

AA = FXAA()
RGB = {"affect_alpha_channel": False}  # the FXAA node on the colour; the alpha gets its own pass here


class FXAAAntiAliasing(Filter):
    """The FXAA node (FXAA 3.11, quality 12) on the input: contrasted luminance edges are smoothed. Affect Alpha
    Channel also anti-aliases the alpha (the coverage of a painted stroke) with its own FXAA pass; off, the alpha
    is kept as it is."""
    type = "paint_fxaa_anti_aliasing"
    label = "FXAA (Anti-Aliasing)"
    catalog = "painter.filter.fxaa_anti_aliasing"
    doc = DOC + "filters/standard-filters/fxaa-anti-aliasing"
    params = {"affect_alpha_channel": B("Affect Alpha Channel", False)}

    def encode(self, ctx, job):
        gray = job.is_gray("output")
        smooth = ctx.scratch(job, gray)
        ctx.dispatch(AA, job, RGB, dsts={"output": smooth})
        alpha = (smooth, gray)  # grayscale or alpha kept: not read
        if job.values["affect_alpha_channel"] and not gray:
            a = ctx.scratch(job, True)
            ctx.dispatch(ALPHA, job, {}, dsts={"output": a})
            alpha = (ctx.scratch(job, True), True)
            ctx.dispatch(AA, job, RGB, srcs={"input": (a, True)}, dsts={"output": alpha[0]})
        ctx.dispatch(FINAL, job, job.values, srcs={"smooth": (smooth, gray), "alpha": alpha})


ALPHA = helper("paint_fxaa_alpha", "fxaa_alpha.wgsl", {"input": Input()}, {"output": Output(PortType.GRAY)},
               main=True)
FINAL = helper("paint_fxaa_final", "fxaa_final.wgsl", {"input": Input(), "smooth": Input(), "alpha": Input()},
               {"output": Output()}, FXAAAntiAliasing.params, main=True)
