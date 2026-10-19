from .common import B, Param, PatternNode, nse, tiling


class Mesh1(PatternNode):
    """Mesh of thin spindles: one soft, narrow blob on every grid point, alternating horizontal and
    vertical in a checker (2 x 2 per tiling unit); Rotate 45 Degrees lays it along the diagonals
    (still tiling)."""

    type = "mesh_thin"
    label = "Mesh Thin"
    catalog = "mesh_1"
    libs = ("simple_lib.wgsl",)
    params = {
        **tiling(2),
        "rotate_45_degrees": Param(B, False, "Rotate 45 Degrees"),
        **nse(),
    }
