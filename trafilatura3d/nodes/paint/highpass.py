from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.paint._lib import DOC, B, F, Filter, blur, helper


class Highpass(Filter):
    """The input minus its blur (Radius), around 0.5."""
    type = "paint_highpass"
    label = "Highpass"
    catalog = "painter.filter.highpass"
    doc = DOC + "filters/standard-filters/highpass"
    params = {"radius": F("Radius", 4.0, 0.0, 32.0), "apply_to_alpha": B("Apply To Alpha", False)}
    unsupported = {"Color Space": "the filter works on the stored values"}

    def encode(self, ctx, job):
        gray = job.is_gray("output")
        blurred = blur(ctx, job, job.values["radius"], True, ctx.scratch(job, gray))
        ctx.dispatch(FINAL, job, job.values, srcs={"blurred": (blurred, gray)})


FINAL = helper("paint_highpass_final", "highpass.wgsl", {"input": Input(), "blurred": Input()},
               {"output": Output()}, Highpass.params, main=True)
