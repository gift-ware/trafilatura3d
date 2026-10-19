from .common import E, F, F2, F3, Input, Param, PatternNode, PortType


class VolumeMask3D(PatternNode):
    """A primitive (Cube, Cylinder with its axis on Z, Sphere) seen through a Position map (unconnected = the
    UV plane, z = 0): the position relative to the centre, rotated (Euler X, Y, Z in
    half-turns), divided by Scale x Size (the half extents), gives a normalised distance d (Cube: the largest
    axis, Sphere: the length, Cylinder: the larger of the radial and axial ones); the mask is (1 - d) /
    Feather Width. The centre is the Position map's value at Position UV, or Position World Space."""

    type = "3d_volume_mask"
    label = "3D Volume Mask"
    catalog = "3d_volume_mask"
    unsupported = {"Position": "the documentation's name; the node has Position World Space (and the UV "
                               "position)"}
    seamless = False
    inputs = {"position": Input(PortType.COLOR, "Position")}
    params = {
        "shape": Param(E, "Cube", "Shape", choices=("Cube", "Cylinder", "Sphere")),
        "scale": Param(F, 0.25, "Scale", 0.0, 1.0),
        "size": Param(F3, [1.0, 1.0, 1.0], "Size", 0.0, 2.0),
        "position_input": Param(E, "UV Position", "Position Input", choices=("UV Position", "World Space Position")),
        "position_uv": Param(F2, [0.5, 0.5], "Position UV", 0.0, 1.0),
        "position_world_space": Param(F3, [0.5, 0.5, 0.5], "Position World Space", 0.0, 1.0),
        "rotation": Param(F3, [0.0, 0.0, 0.0], "Rotation", -1.0, 1.0),
        "feather_width": Param(F, 0.5, "Feather Width", 0.0, 1.0),
    }
