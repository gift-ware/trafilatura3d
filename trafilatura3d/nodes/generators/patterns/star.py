from .common import F, Param, PatternNode, nse


class Star(PatternNode):
    """A solid white five-pointed star (one point up, inner radius 0.382 of the outer, the classic
    pentagram proportions), the outer circle centred, with a half-pixel antialiased edge; Scale = outer
    radius / half the image (1: the top point touches the image edge)."""

    type = "star"
    label = "Star"
    catalog = "star"
    params = {"scale": Param(F, 1.0, "Scale", 0.0, 1.0), **nse()}
