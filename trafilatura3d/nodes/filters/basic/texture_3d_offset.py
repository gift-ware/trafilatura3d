from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class Texture3DOffset(BasicNode):
    """Move the object held by a 3D texture by Offset (fraction of the volume per axis, wrapping).
    The 3D texture is a square image cut into g x g slices (g = 2^round(log2(size) / 3): 16 x 16
    slices of 256 px at 4096, 4 x 4 of 16 px at 64), slice z at column z % g, row z / g. Sampling is
    trilinear inside the volume (never across slice borders)."""

    type = "texture_3d_offset"
    label = "3D Texture Offset"
    category = "Filters/Transforms"
    catalog = "3d_texture_offset"
    inputs = {"input": Input(label="Input")}
    params = {"offset": Param(ParamKind.FLOAT3, [0.0, 0.0, 0.0], "Offset", -1.0, 1.0)}
