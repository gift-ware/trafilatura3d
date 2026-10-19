from .common import E, F2, Param, PatternNode, nse


class GradientRadial(PatternNode):
    """Radial gradient: white at Point 1 (centre), black at the radius |Point 2 - Point 1| and beyond; Shape Cone = linear, Hemisphere = soft and rounded in the centre (sqrt(1 - r^2)). Does not always tile."""

    type = "gradient_radial"
    label = "Gradient Radial"
    catalog = "gradient_radial"
    seamless = False
    params = {
        "shape": Param(E, "Cone", "Shape", choices=("Cone", "Hemisphere")),
        "point_1": Param(F2, [0.5, 0.5], "Point 1", 0.0, 1.0),
        "point_2": Param(F2, [0.8, 0.7], "Point 2", 0.0, 1.0),
        **nse("Non Square Expansion"),
    }
