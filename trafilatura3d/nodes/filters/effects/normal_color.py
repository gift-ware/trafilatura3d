from trafilatura3d.nodes.base import Output, Param, ParamKind, PortType

from ._fx import Effect


class NormalColor(Effect):
    """A uniform normal map colour facing Direction, tilted by Slope Angle (turns): a Uniform Color driven by
    a Direction to normal function (the documentation doesn't list the node): with a = 2 pi Direction and
    s = sin(2 pi Slope Angle), R = 0.5 - 0.5 cos(a) s, G = 0.5 + 0.5 sin(a) s in OpenGL (minus in DirectX), B = 1."""

    type = "normal_color"
    label = "Normal Color"
    category = "Filters/Normal Map"
    outputs = {"output": Output(PortType.COLOR, "Normal")}
    params = {
        "direction": Param(ParamKind.FLOAT, 0.0, "Direction", 0.0, 1.0),
        "slope_angle": Param(ParamKind.FLOAT, 0.0, "Slope Angle", 0.0, 0.25),
        "normal_format": Param(ParamKind.ENUM, "DirectX", "Normal Format", choices=("DirectX", "OpenGL")),
    }
