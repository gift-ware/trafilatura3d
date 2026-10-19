from .common import PatternNode, nse, tiling


class Gaussian2(PatternNode):
    """Gaussian blobs half the size of Gaussian 1's, four per tiling unit on two staggered rows, summed."""

    type = "gaussian_staggered"
    label = "Gaussian Staggered"
    catalog = "gaussian_2"
    libs = ("simple_lib.wgsl",)
    params = {**tiling(1), **nse()}
