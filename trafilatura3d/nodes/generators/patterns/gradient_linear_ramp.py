from .common import E, Param, PatternNode, tiling


class GradientLinear1(PatternNode):
    """Black-to-white linear ramp repeated Tiling times; Rotation turns it clockwise in 90-degree steps (0: white at the top, black at the bottom; 90: white on the right; 180, 270: reversed). A sawtooth: it repeats, but the jump from white back to black also falls on the image border, so it is not seamless."""

    type = "gradient_linear_ramp"
    label = "Gradient Linear Ramp"
    catalog = "gradient_linear_1"
    seamless = False
    params = {
        **tiling(1),
        "rotation": Param(E, "0°", "Rotation", choices=("0°", "90°", "180°", "270°")),
    }
