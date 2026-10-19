from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType


class GrayscaleConversion(Node):
    """gray = dot(RGBA, Channel weights); with Flatten alpha the result is laid over the Background
    value by the input's alpha."""

    type = "grayscale_conversion"
    label = "Grayscale conversion"
    category = "Atomic nodes"
    catalog = "grayscale_conversion"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "channel_weights": Param(ParamKind.FLOAT4, [1 / 3, 1 / 3, 1 / 3, 0.0], "Channel weights", 0.0, 1.0),
        "flatten_alpha": Param(ParamKind.BOOL, False, "Flatten alpha"),
        "background_value": Param(ParamKind.FLOAT, 1.0, "Background value", 0.0, 1.0),
    }
