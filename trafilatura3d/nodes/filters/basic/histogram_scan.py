from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HistogramScan(BasicNode):
    """A sliding levels window: centred on 1 - Position, as wide as fits in 0..1
    (Position 0: all black, 1: white from 1e-4 up, 0.5 with Contrast 0: unchanged), Contrast shrinks it towards
    its centre (1: a step 1e-4 wide, the narrowest window). Invert Position mirrors the position (the window centred
    on Position)."""

    type = "histogram_scan"
    label = "Histogram Scan"
    category = "Filters/Adjustments"
    catalog = "histogram_scan"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "position": Param(ParamKind.FLOAT, 0.0, "Position", 0.0, 1.0),
        "contrast": Param(ParamKind.FLOAT, 0.0, "Contrast", 0.0, 1.0),
        "invert_position": Param(ParamKind.BOOL, False, "Invert Position"),
    }
