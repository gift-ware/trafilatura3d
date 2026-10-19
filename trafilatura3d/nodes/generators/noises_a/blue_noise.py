from trafilatura3d.nodes.base import Param

from .common import F, NoiseNode


class BlueNoiseFast(NoiseNode):
    """Pixel-scale blue noise: per-pixel white noise (tiled on the image size) minus its local mean,
    which removes the low frequencies. Rotation turns the hashing grid: at 1 it is aligned with the
    pixels; away from 1 the grid is rotated (up to 45 degrees at 0), less pixel-scaled, with
    moire waves."""
    type = "blue_noise_fast"
    label = "Blue Noise Fast"
    catalog = "blue_noise_fast"
    params = {"rotation": Param(F, 1.0, "Rotation", 0.0, 1.0)}
