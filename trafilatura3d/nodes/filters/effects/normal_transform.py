from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class NormalTransform(Effect):
    """Transformation 2D for normal maps: the image is resampled through Matrix2x2 about the centre
    plus Offset (Transformation 2D's sampling convention), and the vectors turn with the pixels:
    by the rotation (the angle of its first row) and mirroring of the transposed
    matrix only, or, with Handle Scale and Skew Transformation, by the transposed matrix itself (then
    renormalized)."""

    type = "normal_transform"
    label = "Normal Transform"
    category = "Filters/Normal Map"
    catalog = "normal_transform"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "matrix2x2": Param(ParamKind.FLOAT4, [1.0, 0.0, 0.0, 1.0], "Matrix2x2", -2.0, 2.0),
        "offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Offset", -0.5, 0.5),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
        "handle_scale_and_skew_transformation": Param(ParamKind.BOOL, False, "Handle Scale and Skew Transformation"),
    }
