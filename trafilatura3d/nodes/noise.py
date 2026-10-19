from trafilatura3d.nodes.base import Node, Output, Param, ParamKind, PortType


class Noise(Node):
    """Tileable gradient noise, fractal sum of octaves: a test source (hidden from the library, which
    has its own noises)."""

    type = "noise"
    label = "Noise"
    category = ""
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "scale": Param(ParamKind.INT, 8, "Scale", 1, 64),
        "octaves": Param(ParamKind.INT, 5, "Octaves", 1, 10),
    }
