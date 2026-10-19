from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HistogramScanNonUniform(BasicNode):
    """A histogram scan whose position and contrast can come per pixel from the maps, ours: a linear ramp of slope
    k = 20 (C / (1 - C))^2.8 (fitted on the output) with C = contrast x the Contrast Map (when used) or contrast,
    its centre slid by P from just above white (P 0: all black) to just below black (P 1: all white); P the
    Position Map plus 2 (position - 0.5) when used, else position. Contrast 0.5 with no map is about the input as
    is, 1 a hard step. Unconnected maps read black."""

    type = "histogram_scan_non_uniform"
    label = "Histogram Scan Non-Uniform"
    category = "Filters/Adjustments"
    catalog = "histogram_scan_non_uniform"
    inputs = {"input": Input(PortType.GRAY, "Input"),
              "position_map": Input(PortType.GRAY, "Position Map", (0.0, 0.0, 0.0, 1.0)),
              "contrast_map": Input(PortType.GRAY, "Contrast Map", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "use_position_input": Param(ParamKind.BOOL, True, "Use Position Input"),
        "position": Param(ParamKind.FLOAT, 0.5, "position", 0.0, 1.0),
        "use_contrast_input": Param(ParamKind.BOOL, True, "Use Contrast Input"),
        "contrast": Param(ParamKind.FLOAT, 0.5, "contrast", 0.0, 1.0),
    }
