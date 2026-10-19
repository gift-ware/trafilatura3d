from .common import F, F2, Param, PatternNode, nse


class GradientCircular(PatternNode):
    """Angular gradient revolving around Point 1: black in the direction of Point 2, rising clockwise (on screen) to white just before coming back to it; Angle (turns) is added to the value
    (turns the black line). Never tiles."""

    type = "gradient_circular"
    label = "Gradient Circular"
    catalog = "gradient_circular"
    seamless = False
    params = {
        "point_1": Param(F2, [0.5, 0.5], "Point 1", 0.0, 1.0),
        "point_2": Param(F2, [1.0, 0.5], "Point 2", 0.0, 1.0),
        "angle": Param(F, 0.0, "Angle", 0.0, 1.0),
        **nse("Non Square Expansion"),
    }
