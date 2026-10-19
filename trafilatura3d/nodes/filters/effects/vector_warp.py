from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._fx import Effect


class VectorWarp(Effect):
    """Distorts the input by the vectors of the Vector Map: v = its red and green
    centred on 0.5 (0.5 = no move), each pixel reads the input at uv + v x |v| x Intensity (in UV; long
    vectors move quadratically more); the map needn't be normalised. Vector Format says whether green
    points up (OpenGL) or down (DirectX). Wraps around like the rest of the tiled image."""

    type = "vector_warp"
    label = "Vector Warp"
    category = "Filters/Effects"
    catalog = "vector_warp"
    inputs = {"input": Input(label="Input"),
              "vector_map": Input(PortType.COLOR, "Vector Map", default=(0.5, 0.5, 0.5, 1.0))}
    params = {
        "intensity": Param(ParamKind.FLOAT, 1.0, "Intensity", 0.0, 1.0),
        "vector_format": Param(ParamKind.ENUM, "DirectX", "Vector Format", choices=("DirectX", "OpenGL")),
    }
