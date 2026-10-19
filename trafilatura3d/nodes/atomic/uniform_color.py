from trafilatura3d.nodes.base import Node, Output, Param, ParamKind

COLOR_MODES = ("Grayscale", "Color")


class UniformColor(Node):
    """A flat color (grayscale: its red value). In Color mode alpha is the opacity."""

    type = "uniform_color"
    label = "Uniform color"
    category = "Atomic nodes"
    catalog = "uniform_color"
    outputs = {"output": Output()}
    color_mode = "color_mode"
    params = {
        "color_mode": Param(ParamKind.ENUM, "Color", "Color mode", choices=COLOR_MODES),
        "output_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Output color"),
    }
