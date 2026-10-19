from .common import F2, Param, PatternNode, nse


class GradientAxialReflected(PatternNode):
    """Gradient from Point 1 to Point 2 that starts and ends black and peaks white halfway (a triangle profile along the axis), black beyond the endpoints. Does not tile."""

    type = "gradient_axial_reflected"
    label = "Gradient Axial Reflected"
    catalog = "gradient_axial_reflected"
    seamless = False
    params = {
        "point_1": Param(F2, [0.25, 0.25], "Point 1", 0.0, 1.0),
        "point_2": Param(F2, [0.75, 0.75], "Point 2", 0.0, 1.0),
        **nse("Non Square Expansion"),
    }
