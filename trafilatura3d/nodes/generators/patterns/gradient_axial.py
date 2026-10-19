from .common import F2, Param, PatternNode, nse


class GradientAxial(PatternNode):
    """A straight ramp, 0 at Point 1 and 1 at Point 2, in any direction, flat beyond the two points. Points in uv; with Non Square Expansion distances are measured in square pixels. Does not tile."""

    type = "gradient_axial"
    label = "Gradient Axial"
    catalog = "gradient_axial"
    seamless = False
    params = {
        "point_1": Param(F2, [0.25, 0.25], "Point 1", 0.0, 1.0),
        "point_2": Param(F2, [0.75, 0.75], "Point 2", 0.0, 1.0),
        **nse("Non Square Expansion"),
    }
