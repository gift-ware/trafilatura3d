from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HistogramRange(BasicNode):
    """Squeeze the value range around Position (one Levels on the output range): the output spans
    Position - h .. Position + h, h = min(Range / 2, Position, 1 - Position) (the window centred on Position, as wide
    as fits in 0..1; measured on the original). Range 1 keeps the full range at Position 0.5, Range 0 collapses it
    to Position."""

    type = "histogram_range"
    label = "Histogram Range"
    category = "Filters/Adjustments"
    catalog = "histogram_range"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "range": Param(ParamKind.FLOAT, 0.5, "Range", 0.0, 1.0),
        "position": Param(ParamKind.FLOAT, 0.5, "Position", 0.0, 1.0),
    }
