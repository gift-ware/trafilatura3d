from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode

MODES = ("Greater", "Greater or equal", "Lower", "Lower or equal")


class Threshold(BasicNode):
    """White where the input compares to Threshold as the Mode says, black elsewhere."""

    type = "threshold"
    label = "Threshold"
    category = "Filters/Adjustments"
    catalog = "threshold"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "threshold": Param(ParamKind.FLOAT, 0.01, "Threshold", 0.0, 1.0),
        "mode": Param(ParamKind.ENUM, "Greater", "Mode", choices=MODES),
    }
