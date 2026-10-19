from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HistogramShift(BasicNode):
    """Shift every value by Position, wrapping around past 1: out = in + Position, minus 1 above 1
    (a shift of 1.0 is a full turn, the same as 0)."""

    type = "histogram_shift"
    label = "Histogram Shift"
    category = "Filters/Adjustments"
    catalog = "histogram_shift"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {"position": Param(ParamKind.FLOAT, 0.5, "Position", 0.0, 1.0)}
