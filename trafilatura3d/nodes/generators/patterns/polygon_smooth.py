from .common import B, F, I, Param, PatternNode, nse


class Polygon2(PatternNode):
    """A smooth gradient polygon (a flat edge on the right: the default triangle points left): white at the centre falling to black at the
    edges. Gradient sets how far in the ramp reaches (1 = the whole apothem, towards 0 a solid
    polygon), Curve bends the profile (positive = rounder), Invert Gradient flips it inside the
    polygon. Auto Scale fits the vertices in the image (else the edge midpoints touch it). Tiling repeats it
    Tiling x Tiling times."""

    type = "polygon_smooth"
    label = "Polygon Smooth"
    catalog = "polygon_2"
    params = {
        "tiling": Param(I, 1, "Tiling", 1, 10),
        "sides": Param(I, 3, "Sides", 3, 32),
        "scale": Param(F, 1.0, "Scale", 0.0, 1.0),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0),
        "curve": Param(F, 0.0, "Curve", -1.0, 1.0),
        "gradient": Param(F, 1.0, "Gradient", 0.0, 1.0),
        "invert_gradient": Param(B, False, "Invert Gradient"),
        "auto_scale": Param(B, True, "Auto Scale"),
        **nse(),
    }
