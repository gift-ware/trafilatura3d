"""Legacy noises Perlin noise zoom, 1 and 2, Vertical noise, Dirt Faint and Fur Swirled, all ours, fitted
on the legacy nodes' outputs: Perlin noise zoom (and its presets 1 and 2) on lattice.py, Vertical noise on the Moisture
noise shader, Dirt Faint a sparse convolution (noises_dirt_faint.wgsl), Fur Swirled strands drawn by the FX-Map
engine."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from trafilatura3d.nodes.base import Node, Output, Param, ParamKind, PortType

from ._lib import MAX, F, I, LegacyNode, LegacyNoise, fx_values, levels, mix, stamps, target, xf
from .lattice import blobs
from ..noises_b.noises import MOISTURE_FIT, MoistureNoise1

MOISTURE = MoistureNoise1()


class PerlinNoiseZoom(LegacyNode):
    """Soft blobs (ours, lattice.blobs): on the (2 Distance + 1)^2 lattice of the tile one site per cell at SPREAD x
    u^EDGE steps from its centre (Disorder, in percent, turns them along a circle), a gaussian bell of sigma SIGMA
    steps on each, summed (GAIN), the contrast around the mean raised on small lattices (1 + CONTRAST / n); then the
    preset's Levels (LEVELS). Fitted on the spectrum and histogram of the legacy node's outputs."""
    type = "perlin_noise_zoom"
    label = "Perlin noise zoom (Legacy)"
    params = {"distance": Param(I, 14, "Distance", 0, 64), "disorder": Param(F, 50.0, "Disorder", 0.0, 100.0)}
    DISTANCE: ClassVar[int | None] = None  # a preset's fixed Distance and Disorder (Perlin noise 1 and 2)
    DISORDER = 50.0
    FLIP = False  # mirrored through the centre (Perlin noise 2)
    LEVELS: ClassVar[dict] = {}
    SPREAD, EDGE, SIGMA, GAIN, CONTRAST = 1.55, 0.15, 0.64, 0.1333, 1.0

    def image(self, ctx, job, v, seed, dst=None):
        d = self.DISTANCE if self.DISTANCE is not None else max(int(v["distance"]), 0)
        disorder = self.DISORDER if self.DISTANCE is not None else v["disorder"]
        n = 2 * d + 1
        img = blobs(ctx, job, n, self.SPREAD, self.EDGE, self.SIGMA, self.GAIN, 0.0, 0.01 * disorder,
                    1.0 + self.CONTRAST / n, self.FLIP)
        return levels(ctx, job, img, dst=dst, **self.LEVELS)


class PerlinNoise1(PerlinNoiseZoom):
    """Large soft blobs: Perlin noise zoom with Distance 4 (9 x 9 blobs), then Levels (LEVELS)."""
    type = "perlin_noise_large_legacy"
    label = "Perlin Noise Large (Legacy)"
    params = {}
    DISTANCE = 4
    LEVELS = {"low": 0.043, "high": 0.7}


class PerlinNoise2(PerlinNoiseZoom):
    """Very large soft blobs: Perlin noise zoom with Distance 2 (5 x 5 blobs) mirrored, then Levels (LEVELS)."""
    type = "perlin_noise_huge_legacy"
    label = "Perlin Noise Huge (Legacy)"
    params = {}
    DISTANCE = 2
    FLIP = True
    LEVELS = {"low": 0.094, "high": 0.8, "mid": 0.49}


class VerticalNoise(LegacyNode):
    """Vertical streaks: the Moisture noise shader (noises_b/moisture_noise.wgsl, ours) with its own fit (FIT:
    octaves from SCALE x 4 cells across, Pattern size STRETCH: each bell many times taller than wide) around the gray;
    fitted on the spectrum, histogram and correlations along and across of the legacy node's outputs. Disorder moves
    the streaks along a loop."""
    type = "vertical_noise"
    label = "Vertical noise (Legacy)"
    params = {"disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}
    FIT = {**MOISTURE_FIT, "mo_rough": 0.5, "mo_first": 1.0, "mo_octaves": 4, "mo_gain": 0.2, "mo_gain_up": 0.0,
           "mo_gain_up2": 0.0, "mo_area": 0.0, "mo_cluster_base": 1.0, "mo_cluster": 0.0, "mo_lift": 0.016}
    SCALE, STRETCH = 10, (6.0, 1.0)

    def image(self, ctx, job, v, seed, dst=None):
        values = {**{k: prm.default for k, prm in MOISTURE.params.items()}, **self.FIT,
                  "scale": self.SCALE, "pattern_size": list(self.STRETCH), "disorder": 0.01 * v["disorder"],
                  "mo_seed": float(int(seed) % 9973) * 0.731}
        out = target(ctx, job, dst)
        ctx.dispatch(MOISTURE, job, values, dsts={"output": out})
        return out, True


class _DirtFaint(Node):
    """The faint dirt pass (noises_dirt_faint.wgsl)."""
    type = "noises_dirt_faint"
    registered = False
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"cells": ParamKind.INT, "radius": ParamKind.FLOAT, "weights": ParamKind.FLOAT2,
                         "bias": ParamKind.FLOAT, "gain": ParamKind.FLOAT, "power": ParamKind.FLOAT}

    def wgsl(self) -> str:
        return (Path(__file__).parent / f"{self.type}.wgsl").read_text(encoding="utf-8")


DIRT_FAINT = _DirtFaint()


class DirtFaintLegacy(LegacyNode):
    """Faint bright specks on black (ours): a sparse convolution on three grids (CELLS, 2 CELLS, 4 CELLS cells across,
    WEIGHTS), one tent bump of RADIUS cells per cell at a random point with a random signed height (u - BIAS), the sum
    kept above zero (GAIN, POWER). Fitted on the spectrum, histogram and autocorrelation of the legacy node's
    outputs."""
    type = "dirt_faint_legacy"
    label = "Dirt Faint (Legacy)"
    params = {}
    CELLS, RADIUS, WEIGHTS = 128, 0.6, (1.019, 0.744)
    BIAS, GAIN, POWER = 0.225, 0.481, 0.975

    def image(self, ctx, job, v, seed, dst=None):
        out = target(ctx, job, dst)
        ctx.dispatch(DIRT_FAINT, job, {"cells": self.CELLS, "radius": self.RADIUS, "weights": list(self.WEIGHTS),
                                       "bias": self.BIAS, "gain": self.GAIN, "power": self.POWER},
                     dsts={"output": out})
        return out, True


class FurSwirledLegacy(LegacyNoise):
    """Bright fur strands (ours): COUNT strands at random points, soft-ended bars (Brick, variation SHAPE),
    LENGTH x (1 + u) long and WIDTH x (1 + u) thick (u uniform), each the Max of the ones before. Their direction
    turns across the image, SWIRL x (1 + Disorder^2) x (x + y) turns, and their brightness grows towards the bottom
    right, BRIGHT x u + GLOW x (x + y); then lightened (Max) by its mirror on the anti-diagonal. Fitted on the
    spectrum, histogram and look of the legacy node's outputs, and the same mean and spread from 256 px up. The
    strands' angle and brightness follow the position: the noise does not tile."""
    type = "fur_swirled_legacy"
    label = "Fur Swirled (Legacy)"
    seamless = False
    params = {"disorder": Param(F, 0.7, "Disorder", 0.0, 1.0)}
    FX = fx_values(0.0, 1.0, 0.0)
    COUNT, LENGTH, WIDTH, SHAPE = 1556, 0.2, 0.0035, 1.0
    SWIRL, GLOW, BRIGHT = -0.45, 0.266, 1.05

    def rows(self, v, rng):
        k = self.COUNT
        pos = rng.random((k, 2))
        s = pos.sum(1)
        size = np.stack([self.LENGTH * (1 + rng.random(k)), self.WIDTH * (1 + rng.random(k))], 1)
        turn = self.SWIRL * (1.0 + v["disorder"] ** 2)
        return stamps(pos, size, "Brick", self.BRIGHT * rng.random(k) + self.GLOW * s, 0, MAX, rotation=turn * s,
                      variation=self.SHAPE)

    def finish(self, ctx, job, v, seed, img, dst):
        a = levels(ctx, job, img)  # clamped
        return mix(ctx, job, a, xf(ctx, job, a, (0.0, -1.0, -1.0, 0.0)), "Max", dst=dst)
