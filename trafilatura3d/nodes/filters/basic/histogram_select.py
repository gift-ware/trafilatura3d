from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HistogramSelect(BasicNode):
    """A band select: a tent 1 at Position falling to 0 at Range from it, then Levels
    with input range Contrast / 2 .. 1 - Contrast / 2, so Contrast widens the white core and narrows the
    falloff from both sides (1: a hard band of half-width Range / 2)."""

    type = "histogram_select"
    label = "Histogram Select"
    category = "Filters/Adjustments"
    catalog = "histogram_select"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "position": Param(ParamKind.FLOAT, 0.5, "Position", 0.0, 1.0),
        "range": Param(ParamKind.FLOAT, 0.25, "Range", 0.0, 1.0),
        "contrast": Param(ParamKind.FLOAT, 0.0, "Contrast", 0.0, 1.0),
    }
