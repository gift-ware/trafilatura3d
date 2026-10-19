from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class NormalSobel(Effect):
    """Height map to normal map with Sobel slopes (3x3, smoother than central differences).
    Intensity 1 tilts like the atomic Normal at intensity 10: slopes in height per 25.6 px at 256."""

    type = "normal_sobel"
    label = "Normal Sobel"
    category = "Filters/Normal Map"
    catalog = "normal_sobel"
    libs = ("_height.wgsl",)
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "intensity": Param(ParamKind.FLOAT, 1.0, "Intensity", 0.0, 3.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("OpenGL", "DirectX")),
    }
