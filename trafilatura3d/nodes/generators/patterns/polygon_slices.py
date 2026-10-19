from .common import B, F, F2, I, Param, PatternNode, nse, tiling


class Polygon1(PatternNode):
    """A regular polygon built from Sides triangular slices (centre to edge, a flat edge on the right: a hexagon has a vertex on top),
    repeated Tiling times. Explode pushes the slices outwards; Triangle Size scales each slice about
    its centroid, radially (x) and across (y): only (1, 1) keeps them connected. Slices are solid,
    or with Gradient a ramp from white at the centre to black at the edge (Gradient Invert flips it).
    Auto Scale fits the vertices in the tile (else the edge midpoints touch it); Non Square Tiling
    as in Shape."""

    type = "polygon_slices"
    label = "Polygon Slices"
    catalog = "polygon_1"
    params = {
        "sides": Param(I, 6, "Sides", 3, 32),
        "explode": Param(F, 0.0, "Explode", 0.0, 1.0),
        "triangle_size": Param(F2, [1.0, 1.0], "Triangle Size", 0.0, 1.0),
        "scale": Param(F, 1.0, "Scale", 0.0, 1.0),
        "auto_scale": Param(B, True, "Auto Scale"),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0),
        "gradient": Param(B, False, "Gradient"),
        "gradient_invert": Param(B, False, "Gradient Invert"),
        **tiling(1),
        **nse(),
        "non_square_tiling": Param(B, False, "Non Square Tiling"),  # off by default, as Shape
    }
