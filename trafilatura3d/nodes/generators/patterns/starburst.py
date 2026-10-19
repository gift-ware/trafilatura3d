from .common import F, I, Param, PatternNode, tiling


class Starburst(PatternNode):
    """Branches: thin wedges from the centre of each tile, white at the centre and fading to black
    as they taper to their tip, evenly spread, the first one pointing right (Randomness rotates each
    one randomly around the centre, up to half the spacing). Each branch's length is Size Max, or
    with Size Variation a random value between Size Min and Size Max (1 = the tile edge). Blur
    Amount is a Gaussian blur (1 = sigma 1.6 px at 256 px). Repeated Tiling times."""

    type = "starburst"
    label = "Starburst"
    catalog = "starburst"
    params = {
        "branches": Param(I, 20, "Branches", 3, 512),
        "blur_amount": Param(F, 1.0, "Blur Amount", 0.0, 32.0),
        "randomness": Param(F, 0.0, "Randomness", 0.0, 1.0),
        "size_min": Param(F, 0.5, "Size Min", 0.0, 1.0),
        "size_max": Param(F, 0.86, "Size Max", 0.0, 2.0),
        "size_variation": Param(F, 0.0, "Size Variation", 0.0, 1.0),
        **tiling(1),
    }
