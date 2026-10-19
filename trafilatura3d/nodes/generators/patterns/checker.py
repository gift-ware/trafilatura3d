from .common import B, Param, PatternNode, nse, tiling


class Checker1(PatternNode):
    """Checkerboard, 2 x 2 squares per tiling unit; Rotate 45 Degrees turns it into diamonds along
    the diagonals (still tiling)."""

    type = "checker"
    label = "Checker"
    catalog = "checker_1"
    libs = ("simple_lib.wgsl",)
    params = {
        **tiling(2),
        "rotate_45_degrees": Param(B, False, "Rotate 45 Degrees"),
        **nse(),
    }
