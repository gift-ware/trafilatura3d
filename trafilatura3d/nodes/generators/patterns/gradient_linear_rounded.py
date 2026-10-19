from .common import E, Param, PatternNode, tiling


class GradientLinear2(PatternNode):
    """Double-sided black-white-black gradient with a rounded, pipe-like profile (the parabola 4t(1 - t)), repeated Tiling times; Rotation 0 varies top-bottom, 90 left-right. Seamless."""

    type = "gradient_linear_rounded"
    label = "Gradient Linear Rounded"
    catalog = "gradient_linear_2"
    params = {
        **tiling(1),
        "rotation": Param(E, "0", "Rotation", choices=("0", "90°")),
    }
