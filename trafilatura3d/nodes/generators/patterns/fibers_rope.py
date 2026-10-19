from .common import PatternNode, nse, tiling


class Fibers1(PatternNode):
    """A twisted rope across every tiling unit: four tilted Gaussian strands side by side, for rope,
    mesh or cloth height details."""

    type = "fibers_rope"
    label = "Fibers Rope"
    catalog = "fibers_1"
    libs = ("simple_lib.wgsl",)
    params = {**tiling(1), **nse()}
