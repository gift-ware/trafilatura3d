from trafilatura3d.nodes.base import Input, Output, ParamKind

from ._fx import Effect, prefix_sum
from .flood_fill import F32


class SummedAreaTable(Effect):
    """Summed-area table: each pixel = the sum of the input's pixels above and to the left of it
    (inclusive, from the top left pixel, no wrap-around), divided by the pixel count so the bottom
    right pixel is the image's mean and the values stay in 0-1. Two prefix-sum sweeps on the GPU
    (Hillis-Steele, log2 passes per axis); 32F by default, which it needs."""

    type = "summed_area_table"
    label = "Summed Area Table"
    category = "Filters/Effects"
    catalog = "summed_area_table"
    inputs = {"input": Input(label="Input")}
    outputs = {"output": Output()}
    base_defaults = F32

    def encode(self, ctx, job):
        gray = job.is_gray("output")
        src = ctx.scratch(job, gray)
        ctx.dispatch(FINAL, job, {"pass_": 0}, dsts={"output": src})
        total = prefix_sum(ctx, job, src, gray)
        ctx.dispatch(FINAL, job, {"pass_": 1}, srcs={"input": (total, gray)})


class _Final(Effect):
    """Pass 0: the input divided by the pixel count; pass 1: copy of the sums."""
    type = "fx_summed_area_table_pass"
    registered = False
    inputs = {"input": Input()}
    outputs = SummedAreaTable.outputs
    internal_uniforms = {"pass_": ParamKind.INT}


FINAL = _Final()
