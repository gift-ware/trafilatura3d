from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class Pow(BasicNode):
    """input ^ exponent on every colour channel (alpha kept); negative values keep their sign; default
    exponent 4."""

    type = "pow"
    label = "Pow"
    category = "Filters/Adjustments"
    catalog = "pow"
    inputs = {"input": Input(label="Input")}
    params = {"exponent": Param(ParamKind.FLOAT, 4.0, "exponent", 0.0, 10.0)}
