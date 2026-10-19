from .common import PatternNode, nse, tiling


class Fibers2(PatternNode):
    """Cloth-like ropes: four thin twisted ropes per tiling unit, each made of eight tilted rounded
    strands."""

    type = "fibers_cloth"
    label = "Fibers Cloth"
    catalog = "fibers_2"
    libs = ("simple_lib.wgsl",)
    params = {**tiling(1), **nse()}
