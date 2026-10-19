from .common import B, F, I, Param, PatternNode, nse


class Weave2(PatternNode):
    """Twill of slanted, slightly bent stitches (Scale per row, 2 x Scale staggered rows; Scale
    default 8), each lit from its left: a slow ramp up to the crest, a steep drop on the right. Disorder
    jumbles the stitches (random shift, length and height per stitch, 0-1); Rotate 45 Degrees turns the
    pattern by 45 degrees (still tiling)."""

    type = "weave_twill"
    label = "Weave Twill"
    catalog = "weave_2"
    libs = ("simple_lib.wgsl",)
    unsupported = {"Tiling": "the documentation's older name; the node has Scale (stitches per row)"}
    params = {
        "scale": Param(I, 8, "Scale", 1, 64),
        "disorder": Param(F, 0.0, "Disorder", 0.0, 1.0),
        "rotate_45_degrees": Param(B, False, "Rotate 45 Degrees"),
        **nse(),
    }
