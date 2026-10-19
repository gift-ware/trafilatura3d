from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, GRAY, I, MatNode, Pass, gray_in, passes_of
from trafilatura3d.nodes.materials.scan._common import CATEGORY
from trafilatura3d.nodes.materials.scan._equalize import equalizer_params
from trafilatura3d.nodes.materials.scan.color_equalizer import equalize_sources

N = 8


class MultiColorEqualizer(MatNode):
    """Color Equalizer with the same settings on up to eight inputs (Input Count); each input is
    equalized towards its own average, with its own mask (the Mask Input is shared)."""
    counted_inputs = {"input_count": "input_"}

    type = "multi_color_equalizer"
    label = "Multi Color Equalizer"
    category = CATEGORY
    catalog = "multi_color_equalizer"
    includes = ("_equalize.wgsl",)
    inputs = {**{f"input_{i}": Input(COLOR, f"Input {i}", (0.5, 0.5, 0.5, 1.0)) for i in range(1, N + 1)},
              "mask_input": gray_in("Mask Input", 1.0)}
    outputs = {f"output_{i}": Output(COLOR, f"Output {i}") for i in range(1, N + 1)}
    params = {"input_count": I(1, "Input Count", 1, 8), **equalizer_params()}

    def encode(self, ctx, job):
        ps = passes_of(self, lambda n: [Pass(
            n, f"eq{i}", "multi_color_equalizer.wgsl",
            {f"input_{i}": n.inputs[f"input_{i}"], "blurred": Input(COLOR), "avg": Input(COLOR),
             "m": Input(GRAY)}, {f"output_{i}": n.outputs[f"output_{i}"]},
            subst={"$in": f"input_{i}", "$out": f"output_{i}", "$idx": str(i)}) for i in range(1, N + 1)])
        for i, p in enumerate(ps, 1):
            active = i <= job.values["input_count"]  # inactive inputs: no blur work, black output
            ctx.dispatch(p, job, job.values, srcs=equalize_sources(ctx, job, self, f"input_{i}" if active else ""))
