from trafilatura3d.nodes.base import Input, Node, Param, ParamKind

FILTERING = ("Bilinear | Mipmap", "Bilinear", "Nearest")


class Transformation2D(Node):
    """Resample the input through a 2x2 matrix (rotation, scale, shear) about the centre plus an
    Offset, as sampling: both say where each output pixel samples the input (the matrix column-major;
    2 0 0 2 repeats the image twice, a positive offset moves it left / up). Without tiling, uncovered areas show the
    Matte color. Downscaling averages a footprint of texels (Mipmap mode Auto: from the matrix; Manual: 2^Mipmap
    level)."""

    type = "transformation_2d"
    label = "Transformation 2D"
    category = "Atomic nodes"
    catalog = "transformation_2d"
    inputs = {"input": Input(label="Input")}
    params = {
        "transformation_matrix": Param(ParamKind.FLOAT4, [1.0, 0.0, 0.0, 1.0], "Transformation matrix", -2.0, 2.0),
        "offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Offset", -1.0, 1.0),
        "mipmap_mode": Param(ParamKind.ENUM, "Auto", "Mipmap mode", choices=("Auto", "Manual")),
        "mipmap_level": Param(ParamKind.INT, 0, "Mipmap level", 0, 12),
        "matte_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 0.0], "Matte color"),
        "filtering": Param(ParamKind.ENUM, "Bilinear | Mipmap", "Filtering", choices=FILTERING),
    }
