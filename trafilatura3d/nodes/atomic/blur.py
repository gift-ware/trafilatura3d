from trafilatura3d.nodes.base import Input, Param, ParamKind

from .warp import PixelSized


class Blur(PixelSized):
    """Box blur (uniform average over a neighbourhood), separable: a horizontal pass into a scratch
    texture, then a vertical one. Intensity is in pixels of a 256 px image, so the look holds at
    every output size, times Pixel Size per axis (library nodes set it: Fractal Sum
    Base blurs its octaves at Intensity 1 and Pixel Size 64 ... 1/16, half a cell of each)."""

    type = "blur"
    label = "Blur"
    category = "Atomic nodes"
    catalog = "blur"
    inputs = {"input": Input(label="Input")}
    params = {"intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 64.0)}
    internal_uniforms = {**PixelSized.internal_uniforms, "vertical": ParamKind.INT}

    def encode(self, ctx, job):
        gray = job.is_gray("output")
        scratch = ctx.scratch(job, gray)
        v = {**job.values, "pixel_size_": job.extra.get("pixel_size", (1.0, 1.0))}
        ctx.dispatch(self, job, {**v, "vertical": 0}, dsts={"output": scratch})
        ctx.dispatch(self, job, {**v, "vertical": 1}, srcs={"input": (scratch, gray)})
