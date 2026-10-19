from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import COPY, BasicNode, source

COUNT = 20


class MultiSwitch(BasicNode):
    """Pass through the input chosen by Input Selection (1-based), unmodified. Input Number
    only shows the inputs: a selection past it gives that (hidden, so empty) input. Only the selected input is
    bound, so all 20 inputs fit any GPU."""
    counted_inputs = {"input_number": "input_"}

    type = "multi_switch"
    label = "Multi Switch"
    category = "Filters/Blending"
    catalog = "multi_switch"
    inputs = {f"input_{i}": Input(label=f"Input {i}", default=(0.0, 0.0, 0.0, 1.0)) for i in range(1, COUNT + 1)}
    params = {
        "input_number": Param(ParamKind.INT, 2, "Input Number", 2, COUNT),
        "input_selection": Param(ParamKind.INT, 1, "Input Selection", 1, COUNT),
    }

    def encode(self, ctx, job):
        sel = max(1, min(COUNT, job.values["input_selection"]))
        ctx.dispatch(COPY, job, {}, srcs={"input": source(ctx, job, f"input_{sel}")})
