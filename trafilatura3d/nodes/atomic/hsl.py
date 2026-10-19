from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType


class HSL(Node):
    """Hue, saturation and lightness: each slider adds 2 (value - 0.5) to its component; 0.5 leaves the image
    unchanged."""

    type = "hsl"
    label = "HSL"
    category = "Atomic nodes"
    catalog = "hsl"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "hue": Param(ParamKind.FLOAT, 0.5, "Hue", 0.0, 1.0),
        "saturation": Param(ParamKind.FLOAT, 0.5, "Saturation", 0.0, 1.0),
        "lightness": Param(ParamKind.FLOAT, 0.5, "Lightness", 0.0, 1.0),
    }
