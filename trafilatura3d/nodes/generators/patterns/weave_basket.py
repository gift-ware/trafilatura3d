from .common import B, Param, PatternNode, nse, tiling


class Weave1(PatternNode):
    """Basket weave: blocks of five parallel rounded threads, vertical and horizontal blocks
    alternating in a checker, 4 x 4 blocks repeated by Tiling in steps (2^min(int(sqrt(Tiling)), 3) times, once
    at Tiling 1); Rotate 45 Degrees lays it along the diagonals (still tiling)."""

    type = "weave_basket"
    label = "Weave Basket"
    catalog = "weave_1"
    libs = ("simple_lib.wgsl",)
    params = {
        **tiling(1),
        "rotate_45_degrees": Param(B, False, "Rotate 45 Degrees"),
        **nse(),
    }
