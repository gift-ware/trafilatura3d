from .common import PatternNode, nse, tiling


class Gaussian1(PatternNode):
    """Soft Gaussian blobs at the corners and the centre of every tiling unit (a diagonal grid), summed."""

    type = "gaussian_diagonal"
    label = "Gaussian Diagonal"
    catalog = "gaussian_1"
    libs = ("simple_lib.wgsl",)
    params = {**tiling(1), **nse()}
