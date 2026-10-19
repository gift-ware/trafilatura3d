from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class QuantizeGrayscale(BasicNode):
    """The grayscale twin of Quantize Color (Simple) (the documentation doesn't list
    it): round to Quantize evenly spaced levels including 0 and 1,
    round(v * (N - 1)) / (N - 1) (Levels out high (N - 1) / 255 at 8 bits, then in high the same)."""

    type = "quantize_grayscale"
    label = "Quantize Grayscale"
    category = "Filters/Adjustments"
    inputs = {"input": Input(PortType.GRAY, "Grayscale")}
    outputs = {"output": Output(PortType.GRAY, "Output")}
    params = {"quantize": Param(ParamKind.INT, 3, "Quantize", 2, 256)}
