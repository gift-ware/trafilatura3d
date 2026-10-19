from .common import B, Param, PatternNode, nse, tiling


class Mesh2(PatternNode):
    """Mesh of fat blocks: one wide rounded capsule on every grid point, alternating horizontal and
    vertical in a checker (2 x 2 per tiling unit), merging into a woven look; Rotate 45 Degrees lays
    it along the diagonals (still tiling)."""

    type = "mesh_thick"
    label = "Mesh Thick"
    catalog = "mesh_2"
    libs = ("simple_lib.wgsl",)
    params = {
        **tiling(2),
        "rotate_45_degrees": Param(B, False, "Rotate 45 Degrees"),
        **nse(),
    }
