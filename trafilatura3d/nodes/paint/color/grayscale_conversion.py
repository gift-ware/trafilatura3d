from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, B, E, F, Filter

MODES = ("Desaturation", "Luma", "Average", "Max", "Min", "Channels Weights")


class GrayscaleConversion(Filter):
    """Grayscale of the layer input or of the Custom Input (Channel Input). Mode: Desaturation (halfway between the
    strongest and weakest channel), Luma, Average, Max, Min, or Channels Weights = R*Red + G*Green + B*Blue +
    A*Alpha (not normalized, clamped to 0-1). Then Balance (0.5 = unchanged), Contrast (0 = unchanged), Invert."""
    type = "paint_grayscale_conversion"
    label = "Grayscale Conversion"
    catalog = "painter.filter.grayscale_conversion"
    doc = DOC + "filters/standard-filters/grayscale-conversion"
    includes = ("color/color",)
    inputs = {"input": Input(label="Input"),
              "custom_input": Input(PortType.COLOR, "Custom Input: Color", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {"channel_input": E("Channel Input", ("Layer Input", "Custom Input")), "invert": B("Invert", False),
              "balance": F("Balance", 0.5, 0.0, 1.0), "contrast": F("Contrast", 0.0, 0.0, 1.0),
              "mode": E("Mode", MODES, "Channels Weights"),
              "red": F("Red", 1.0 / 3.0, 0.0, 1.0), "green": F("Green", 1.0 / 3.0, 0.0, 1.0),
              "blue": F("Blue", 1.0 / 3.0, 0.0, 1.0), "alpha": F("Alpha", 0.0, 0.0, 1.0)}
