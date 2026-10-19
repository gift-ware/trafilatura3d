from .common import E, F, Param, PatternNode, tiling


class GradientLinear3(PatternNode):
    """Double-sided gradient with sharp straight slopes: black at the tile edges, white at the peak placed by Position, repeated Tiling times; Rotation 0 varies top-bottom (Position from the top), 90 left-right. Seamless."""

    type = "gradient_linear_peak"
    label = "Gradient Linear Peak"
    catalog = "gradient_linear_3"
    params = {
        **tiling(1),
        "position": Param(F, 0.5, "Position", 0.0, 1.0),
        "rotation": Param(E, "0", "Rotation", choices=("0", "90°")),
    }
