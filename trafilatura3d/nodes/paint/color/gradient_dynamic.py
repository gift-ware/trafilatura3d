from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, E, F, Filter


class GradientDynamic(Filter):
    """The input's grayscale (luminance of a color input) picks a color along one line of the Gradient Source image:
    Horizontal reads the row at Gradient Input Position (x = the value), Vertical the column (y = the value).
    Without a Gradient Source the value itself is written (a black to white gradient)."""
    type = "paint_gradient_dynamic"
    label = "Gradient Dynamic"
    catalog = "painter.filter.gradient_dynamic"
    doc = DOC + "filters/standard-filters/gradient-dynamic"
    inputs = {"input": Input(label="Input"),
              "gradient_source": Input(PortType.COLOR, "Gradient Source: Color", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"gradient_orientation": E("Gradient Orientation", ("Horizontal", "Vertical")),
              "gradient_input_position": F("Gradient Input Position", 0.5, 0.0, 1.0)}
