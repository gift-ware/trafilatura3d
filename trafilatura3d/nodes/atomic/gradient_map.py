from trafilatura3d.nodes.base import Input, Node, Param, ParamKind, PortType

DEFAULT = {"interpolation": "linear", "keys": [{"pos": 0.0, "color": [0, 0, 0, 1]}, {"pos": 1.0, "color": [1, 1, 1, 1]}]}


class GradientMap(Node):
    """Map the grayscale input through a gradient (Grayscale mode: the gradient's luminance).
    Values out of [0, 1] repeat or clamp per Gradient adressing."""

    type = "gradient_map"
    label = "Gradient map"
    category = "Atomic nodes"
    catalog = "gradient_map"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    color_mode = "color_mode"
    params = {
        "color_mode": Param(ParamKind.ENUM, "Color", "Color mode", choices=("Color", "Grayscale")),
        "gradient_adressing": Param(ParamKind.ENUM, "Clamp", "Gradient adressing", choices=("Repeat", "Clamp")),
        "gradient": Param(ParamKind.GRADIENT, DEFAULT, "Gradient"),
    }
