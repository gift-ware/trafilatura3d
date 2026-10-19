from .common import B, E, F, F2, Param, PatternNode, SHAPES, nse, tiling


class Shape(PatternNode):
    """One built-in shape per tile, repeated Tiling times: Scale (uniform), Size (non-uniform), Angle in turns;
    Rotation 45 turns the image and shrinks it by 1/sqrt(2) (a 45 degree lattice of shapes). Non Square Expansion
    keeps the shape square on non-square images; Non Square Tiling then repeats it along the longer axis (else one
    row of shapes is centred and the rest is black). Defaults: Scale 1, Non Square Tiling off."""

    type = "shape"
    label = "Shape"
    catalog = "shape"
    params = {
        **tiling(1),
        "pattern": Param(E, "Square", "Pattern", choices=SHAPES),
        "pattern_specific": Param(F, 0.0, "Pattern Specific", 0.0, 1.0),
        "scale": Param(F, 1.0, "Scale", 0.0, 1.0),
        "size": Param(F2, [1.0, 1.0], "Size", 0.0, 1.0),
        "angle": Param(F, 0.0, "Angle", 0.0, 1.0),
        "rotation_45": Param(B, False, "Rotation 45°"),
        **nse(),
        "non_square_tiling": Param(B, False, "Non Square Tiling"),
    }
