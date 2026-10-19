from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class Curvature(Effect):
    """Single-pass curvature of a normal map: the divergence of the normals' XY between the direct
    neighbours (one pixel, so lines stay pixel-thin), 0.5 + divergence x 0.4 x Intensity: convex white,
    concave black, flat 50% gray."""

    type = "curvature"
    label = "Curvature (Filter Node)"
    category = "Filters/Effects"
    catalog = "curvature"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "intensity": Param(ParamKind.FLOAT, 1.0, "Intensity", 0.0, 10.0),
        "normal_format": Param(ParamKind.ENUM, "DirectX", "Normal Format", choices=("DirectX", "OpenGL")),
    }
