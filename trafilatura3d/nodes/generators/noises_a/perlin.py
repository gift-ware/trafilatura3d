"""Perlin noise (perlin_noise.wgsl): gradient noise built from surflets, one random gradient per cell of the Scale
grid, its spectrum and spread fitted on the output of the earlier implementation."""

from .common import CLASSIC, NoiseNode, scale


class PerlinNoise(NoiseNode):
    """Smooth gradient noise on the Scale x Scale grid; Disorder turns the gradients (1 loops). Scale 32 (the
    documentation's preview)."""
    type = "perlin_noise"
    label = "Perlin noise"
    catalog = "perlin_noise"
    params = {**scale(32, 1, 64), **{k: v for k, v in CLASSIC.items() if k not in (
        "disorder_anisotropy", "disorder_anisotropy_angle")}}
