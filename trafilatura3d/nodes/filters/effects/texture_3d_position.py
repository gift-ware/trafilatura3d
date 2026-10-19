from trafilatura3d.nodes.base import Output, Param, ParamKind, PortType

from ._fx import Effect


class Texture3DPosition(Effect):
    """Position slices of the unit cube as a 3D texture atlas: Grid Size columns x
    rows of slices (16x16 = 256 slices, left to right, top to bottom); in each slice X (red) and Y
    (green, downwards) run across it at the pixel centres, and blue is the slice's Z = index / slices.
    Fed to 3D noises it bakes them into a 3D texture."""

    type = "3d_texture_position"
    label = "3D Texture Position"
    category = "Filters/Effects"
    catalog = "3d_texture_position"
    libs = ("_volume.wgsl", "_volume_grid.wgsl")
    seamless = False
    outputs = {"output": Output(PortType.COLOR)}
    params = {"grid_size": Param(ParamKind.INT2, [16, 16], "Grid Size", 0, 16)}
