from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, E, F
from trafilatura3d.nodes.materials.hdri._common import HdriNode


class ColorTemperatureAdjustment(HdriNode):
    """White balance: multiplies the image by the ratio between a blackbody tint and 6500 K white
    (Temperature: + warmer, - colder) and shifts green against magenta, at constant luminance.
    LDR (sRGB) inputs are linearized for the adjustment and re-encoded."""

    type = "color_temperature_adjustment"
    label = "Color Temperature Adjustment"
    catalog = "color_temperature_adjustment"
    inputs = {"input": Input(COLOR, "Input", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    params = {
        "temperature": F(0.0, "Temperature", -1.0, 1.0),
        "magenta_green": F(0.0, "Magenta-Green", -1.0, 1.0),
        "color_space": E("HDR (linear)", "Color Space", ("HDR (linear)", "LDR (sRGB)")),
    }
