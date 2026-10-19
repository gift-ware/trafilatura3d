from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HDRRangeViewer(BasicNode):
    """Debug view of values outside [0, 1]: in-range pixels as dimmed gray, values above 1 from red
    (just above) to yellow (Range Max), values below 0 from blue (just below) to cyan (Range Min)."""

    type = "hdr_range_viewer"
    label = "HDR Range Viewer"
    category = "Filters/Adjustments"
    catalog = "hdr_range_viewer"
    inputs = {"input": Input(label="Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "range_min": Param(ParamKind.FLOAT, -1.0, "Range Min", -2.0, 0.0),
        "range_max": Param(ParamKind.FLOAT, 2.0, "Range Max", 1.0, 3.0),
    }
