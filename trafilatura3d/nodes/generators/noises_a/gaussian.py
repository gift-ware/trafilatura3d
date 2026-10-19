"""Gaussian noise and Gaussian spots 1 (gaussian.wgsl), both ours. Gaussian noise is a sparse convolution noise:
one Gaussian bell per cell of the Scale grid at a jittered point, summed (a smooth noise, dark mean, bell-shaped
histogram), its width, mean and spread fitted. Gaussian spots 1: narrow Gaussian bells on
a sub-cell grid, shifted down and clamped into spots on black, mean and spread matched on the preview."""

from .common import CLASSIC, NoiseNode, scale


class GaussianNoise(NoiseNode):
    """A smooth noise made of Gaussian patterns."""
    type = "gaussian_noise"
    label = "Gaussian noise"
    catalog = "gaussian_noise"
    shader = "gaussian"
    params = {**scale(32), **CLASSIC}
    variant = 0


class GaussianSpots1(GaussianNoise):
    """Soft spots on black: narrow Gaussian bells, shifted down and clamped."""
    type = "gaussian_spots_large"
    label = "Gaussian Spots Large"
    catalog = "gaussian_spots_1"
    params = {**scale(1, 1, 16), **CLASSIC}
    scale_cells = 8
    variant = 1
