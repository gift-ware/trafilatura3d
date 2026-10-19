from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class CurvatureSobel(Effect):
    """Curvature of a normal map with Sobel derivatives of the normals' XY (3x3, thicker and cleaner
    lines than Curvature): 0.5 + (Sobel X of the red + Sobel Y of the green, encoded
    0-1) x Intensity / 2, convex white, concave black."""

    type = "curvature_sobel"
    label = "Curvature Sobel"
    category = "Filters/Effects"
    catalog = "curvature_sobel"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "intensity": Param(ParamKind.FLOAT, 0.5, "Intensity", 0.0, 1.0),
        "input_normal_type": Param(ParamKind.ENUM, "OpenGL", "Normal type", choices=("DirectX", "OpenGL")),
    }
