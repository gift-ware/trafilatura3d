from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType


class Normal(Node):
    """Tangent-space normal map from a height map: slopes (Sobel) scaled by Intensity, encoded to
    RGB; the green channel follows the Normal format."""

    type = "normal"
    label = "Normal"
    category = "Atomic nodes"
    catalog = "normal"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 100.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal format", choices=("DirectX", "OpenGL")),
        "alpha_channel_content": Param(ParamKind.ENUM, "Force Alpha To 1", "Alpha channel content",
                                       choices=("Fill Alpha With Input", "Force Alpha To 1")),
    }
