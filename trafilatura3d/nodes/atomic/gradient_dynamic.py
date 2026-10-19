from trafilatura3d.nodes.base import Input, Node, Param, ParamKind, PortType


class GradientDynamic(Node):
    """Use a row (Horizontal) or column (Vertical) of the Gradient input as the gradient that maps
    the Grayscale input; the row/column is chosen by Gradient input position."""

    type = "gradient_dynamic"
    label = "Gradient (Dynamic)"
    category = "Atomic nodes"
    catalog = "gradient_dynamic"
    inputs = {"grayscale_input": Input(PortType.GRAY, "Grayscale input"),
              "gradient_input": Input(label="Gradient input")}

    def output_types(self, input_types, values):  # follows the Gradient input only
        return {"output": input_types["gradient_input"] or PortType.GRAY}

    params = {
        "gradient_addressing": Param(ParamKind.ENUM, "Clamp", "Gradient addressing", choices=("Repeat", "Clamp")),
        "gradient_orientation": Param(ParamKind.ENUM, "Horizontal", "Gradient orientation",
                                      choices=("Horizontal", "Vertical")),
        "gradient_input_position": Param(ParamKind.FLOAT, 0.5, "Gradient input position", 0.0, 1.0),
    }
