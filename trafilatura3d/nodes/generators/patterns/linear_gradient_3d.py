from .common import E, F, F2, F3, Input, Param, PatternNode, PortType


class LinearGradient3D(PatternNode):
    """A ramp in 3D, 0 at Point 1 and 1 at Point 2, evaluated per pixel from a Position
    map (unconnected = the UV plane, z = 0), projected in 3D. UV Positions: the
    points are the Position map's values at the two UV points (Point 1/2 UV); World Space Positions: the
    points themselves. Contrast sharpens the ramp around its middle (1 = a hard step)."""

    type = "3d_linear_gradient"
    label = "3D Linear Gradient"
    catalog = "3d_linear_gradient"
    seamless = False
    inputs = {"position": Input(PortType.COLOR, "Position")}
    params = {
        "points_position_mode": Param(E, "UV Positions", "Points Position Mode",
                                      choices=("UV Positions", "World Space Positions")),
        "point_1_uv": Param(F2, [0.75, 0.75], "Point 1", 0.0, 1.0),
        "point_2_uv": Param(F2, [0.25, 0.25], "Point 2", 0.0, 1.0),
        "point_1_world": Param(F3, [0.0, 0.0, 0.0], "Point 1 (World)", 0.0, 1.0),
        "point_2_world": Param(F3, [1.0, 1.0, 1.0], "Point 2 (World)", 0.0, 1.0),
        "contrast": Param(F, 0.0, "Contrast", 0.0, 1.0),
    }
