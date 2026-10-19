"""Fractal sum base and its presets Fractal sum 1-4: levels Min. level .. Max. level of random signed
bells, one per cell (level L has 2^L cells across the tile, so levels are resolution independent;
levels finer than a pixel add per-pixel grain), level L+1 weighted Roughness times level L and the
weights summing to 1 (a `roughness` function; 1: every level as strong), the sum around
mid-grey stretched by Contrast, then times Global opacity (above 1 burns to white). The presets are the base with fixed values (`fractal_sum.wgsl`)."""

from trafilatura3d.nodes.base import Param

from .common import DISORDER, NSE, F, I, NoiseNode


class FractalSumBase(NoiseNode):
    """Fractal noise whose first and last octave and their relative weights are parameters."""
    type = "fractal_sum_base"
    label = "Fractal sum base"
    catalog = "fractal_sum_base"
    params = {
        "roughness": Param(F, 1.0, "Roughness", 0.0, 2.0),
        "min_level": Param(I, 1, "Min. level", 1, 12),
        "max_level": Param(I, 11, "Max. level", 1, 12),
        **DISORDER,
        "contrast": Param(F, 0.0, "Contrast", 0.0, 1.0),
        "global_opacity": Param(F, 1.0, "Global opacity", 0.0, 2.0),
        **NSE,
    }


class FractalSumPreset(NoiseNode):
    shader = "fractal_sum"
    params = {**DISORDER, **NSE}


class FractalSum1(FractalSumPreset):
    """Fractal sum preset: all levels, finer ones a little stronger, darker."""
    type = "fractal_sum_full"
    label = "Fractal Sum Full"
    catalog = "fractal_sum_1"
    variant = 0


class FractalSum2(FractalSumPreset):
    """Fractal sum preset: all levels, the fine grain dominant, dark and contrasted."""
    type = "fractal_sum_gritty"
    label = "Fractal Sum Gritty"
    catalog = "fractal_sum_2"
    variant = 1


class FractalSum3(FractalSumPreset):
    """Fractal sum preset: small round blobs (levels from 6, smooth)."""
    type = "fractal_sum_dotted"
    label = "Fractal Sum Dotted"
    catalog = "fractal_sum_3"
    variant = 2


class FractalSum4(FractalSumPreset):
    """Fractal sum preset: bigger round blobs (levels from 5, smooth), brighter."""
    type = "fractal_sum_blobby"
    label = "Fractal Sum Blobby"
    catalog = "fractal_sum_4"
    variant = 3
