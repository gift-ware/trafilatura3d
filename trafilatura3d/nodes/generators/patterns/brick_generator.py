from .common import B, F2, F4, Param, ParamKind, PatternNode, nse


class BrickGenerator(PatternNode):
    """Running-bond bricks, with vector parameters. Bricks = per row (x) and rows (y). Bevel:
    x / y = width of the edge slope across the vertical / horizontal joints (relative to the brick height on
    both axes with Keep Ratio, else to each side), z = falloff (0 linear, 1 smooth), w = corner rounding (the
    two slopes multiplied instead of their minimum). Gap: x / y = vertical / horizontal joints. Middle Size:
    x / y = width / height share of every other column / row (0.5 = equal). Offset: x = shift of odd rows
    (fraction of a brick), y = random shift per row. Per brick, deterministic randoms: Height x / y = the two
    levels a brick takes (z: the share at y, 0 = half), w = luminance variation (negative: inverted random);
    Slope x / y = tilt along / across the brick (sign: direction), z = more bricks tilted along x, w = more
    bricks flipped. Default look of the documentation preview (4 x 4 bricks)."""

    type = "brick_generator"
    label = "Brick Generator"
    catalog = "brick_generator"
    libs = ("simple_lib.wgsl",)
    params = {
        "bricks": Param(ParamKind.INT2, [4, 4], "Bricks", 1, 64),
        "bevel": Param(F4, [0.5, 0.5, 0.0, 0.0], "Bevel", 0.0, 1.0),
        "keep_ratio": Param(B, True, "Keep Ratio"),
        "gap": Param(F2, [0.0, 0.0], "Gap", 0.0, 1.0),
        "middle_size": Param(F2, [0.5, 0.5], "Middle Size", 0.0, 1.0),
        "height": Param(F4, [1.0, 1.0, 0.0, 0.0], "Height", -1.0, 1.0),
        "slope": Param(F4, [0.0, 0.0, 0.0, 0.0], "Slope", -1.0, 1.0),
        "offset": Param(F2, [0.5, 0.0], "Offset", 0.0, 1.0),
        **nse(),
    }
