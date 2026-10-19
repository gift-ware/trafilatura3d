from .common import F, I, Param, PatternNode, nse


class Brick2(PatternNode):
    """Bricks on 2 x Scale rows (Scale default 2), alternating one long brick (the unit width)
    and two half-unit bricks whose joints sit at 45/256 and 173/256 of the unit: Interstice Width = joint
    width (share of the row height), Edge Smoothness from a hard edge (the default) to a soft bevel."""

    type = "brick_mixed"
    label = "Brick Mixed"
    catalog = "brick_2"
    libs = ("simple_lib.wgsl",)
    unsupported = {"Tiling": "the documentation's older name; the node has Scale (pairs of rows)"}
    params = {
        "scale": Param(I, 2, "Scale", 1, 16),
        "edge_smoothness": Param(F, 0.0, "Edge Smoothness", 0.0, 1.0),
        "interstice_width": Param(F, 0.09375, "Interstice Width", 0.0, 1.0),
        **nse(),
    }
