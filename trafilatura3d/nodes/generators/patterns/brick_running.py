from .common import F, I, Param, PatternNode, nse


class Brick1(PatternNode):
    """Running-bond bricks (2:1, Scale rows of Scale / 2 bricks, odd rows shifted half a brick;
    Scale default 4): Interstice Width = joint width (share of the brick height), Edge Smoothness from a hard
    edge (the default) to a soft bevel."""

    type = "brick_running"
    label = "Brick Running"
    catalog = "brick_1"
    libs = ("simple_lib.wgsl",)
    unsupported = {"Tiling": "the documentation's older name; the node has Scale (rows)"}
    params = {
        "scale": Param(I, 4, "Scale", 1, 16),
        "edge_smoothness": Param(F, 0.0, "Edge Smoothness", 0.0, 1.0),
        "interstice_width": Param(F, 0.09375, "Interstice Width", 0.0, 1.0),
        **nse(),
    }
