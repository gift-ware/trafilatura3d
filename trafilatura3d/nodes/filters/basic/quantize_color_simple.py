from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class QuantizeColorSimple(BasicNode):
    """Round each channel to N evenly spaced levels including 0 and 1 (N = the channel's step
    count): round(v * (N - 1)) / (N - 1). Grayscale inputs use the Red channel's count."""

    type = "quantize_color_simple"
    label = "Quantize Color (Simple)"
    category = "Filters/Adjustments"
    catalog = "quantize_color_simple"
    inputs = {"input": Input(label="Input")}
    params = {
        "quantize_red_channel": Param(ParamKind.INT, 4, "Quantize Red Channel", 2, 256),
        "quantize_green_channel": Param(ParamKind.INT, 4, "Quantize Green Channel", 2, 256),
        "quantize_blue_channel": Param(ParamKind.INT, 4, "Quantize Blue Channel", 2, 256),
        "quantize_alpha_channel": Param(ParamKind.INT, 4, "Quantize Alpha Channel", 2, 256),
    }
