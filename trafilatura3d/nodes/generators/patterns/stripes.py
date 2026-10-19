from .common import B, E, F, I, Param, PatternNode, nse


class Stripes(PatternNode):
    """Stripes stripes across the image (vertical at Shift 0); Shift leans them like a backslash by
    whole stripes over the image height so they always tile, around the top edge (Align Edges) or the middle
    (Center). Width = fraction of the period that is white, Softness = the edge ramp as a fraction
    of the half width. Filtering antialiases the edges (off: hard pixels)."""

    type = "stripes"
    label = "Stripes"
    catalog = "stripes"
    params = {
        "stripes": Param(I, 10, "Stripes", 1, 100),
        "width": Param(F, 0.5, "Width", 0.0, 1.0),
        "softness": Param(F, 0.0, "Softness", 0.0, 1.0),
        "shift": Param(I, 10, "Shift", 0, 20),
        "align": Param(E, "Edges", "Align", choices=("Edges", "Center")),
        "filtering": Param(B, False, "Filtering"),
        **nse(),
    }
