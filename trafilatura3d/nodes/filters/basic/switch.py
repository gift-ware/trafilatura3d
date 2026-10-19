from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import COPY, BasicNode, source


class Switch(BasicNode):
    """Input 1 when Switch is on, else Input 2, unmodified."""

    type = "switch"
    label = "Switch"
    category = "Filters/Blending"
    catalog = "switch"
    inputs = {"input_1": Input(label="Input 1 (True)", default=(0.0, 0.0, 0.0, 1.0)),
              "input_2": Input(label="Input 2 (False)", default=(0.0, 0.0, 0.0, 1.0))}
    params = {"switch_value": Param(ParamKind.BOOL, True, "Switch")}

    def encode(self, ctx, job):
        port = "input_1" if job.values["switch_value"] else "input_2"
        ctx.dispatch(COPY, job, {}, srcs={"input": source(ctx, job, port)})
