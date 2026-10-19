from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class Invert(BasicNode):
    """1 - input on every colour channel when Invert is on (alpha kept), else a pass-through."""

    type = "invert"
    label = "Invert"
    category = "Filters/Adjustments"
    catalog = "invert"
    inputs = {"input": Input(label="Input")}
    params = {"invert": Param(ParamKind.BOOL, True, "Invert")}
