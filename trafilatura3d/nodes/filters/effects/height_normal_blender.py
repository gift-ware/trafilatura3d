from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect

FLAT = (0.5, 0.5, 1.0, 1.0)


class HeightNormalBlender(Effect):
    """Converts the Height input to a normal map (Sobel, Normal Intensity as in Normal Sobel) and lays
    it on the Normal input with reoriented normal mapping, so both reliefs stay correct."""

    type = "height_normal_blender"
    label = "Height Normal Blender"
    category = "Filters/Normal Map"
    catalog = "height_normal_blender"
    libs = ("_height.wgsl",)
    inputs = {"input": Input(PortType.GRAY, "Height"), "normal": Input(PortType.COLOR, "Normal", default=FLAT)}
    main_input = "normal"
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "normal_intensity": Param(ParamKind.FLOAT, 0.0, "Normal Intensity", 0.0, 16.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
    }
