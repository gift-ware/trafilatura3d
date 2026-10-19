from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class HeightToNormalWorldUnits(Effect):
    """Height map to normal map in real units: the image spans Surface Size (cm) and its 0-1 range is
    Height Depth (cm), so the slope is dh/du x depth / size (defaults 300 and 16 cm).
    Standard sampling = central differences, Sobel = 3x3 Sobel."""

    type = "height_to_normal_world_units"
    label = "Height to Normal World Units"
    category = "Filters/Normal Map"
    catalog = "height_to_normal_world_units"
    libs = ("_height.wgsl",)
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "surface_size_cm": Param(ParamKind.FLOAT, 300.0, "Surface Size (cm)", 0.0, 1000.0),
        "height_depth_cm": Param(ParamKind.FLOAT, 16.0, "Height Depth (cm)", 0.0, 100.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("OpenGL", "DirectX")),
        "sampling": Param(ParamKind.ENUM, "Standard", "Sampling", choices=("Standard", "Sobel")),
    }
