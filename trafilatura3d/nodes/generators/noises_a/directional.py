"""Directional noises (1-4), Directional scratches and Anisotropic noise.

Directional noise 1 (Fibrous): gradient-noise octaves averaged along the direction (directional_fibrous.wgsl).
Directional noise 2-4: value-noise octaves and a pixel grain box-blurred along the direction (Angle,
varied by Angle random through a coarse noise): streaks at any angle that still tile, since every
sample comes from a periodic lattice. Scratches: long thin Cones scattered at random, overlaid with Max. Anisotropic
noise: strips of random values, hard across, fading along from a random phase."""

import math

import numpy as np

from trafilatura3d.fxmap import BLENDING, PATTERNS
from trafilatura3d.nodes.atomic.stamp_map import ROW, draw_patterns
from trafilatura3d.nodes.base import Node, Param

from .common import ANISO, CATEGORY, DISORDER, GRAY_OUT, NSE, TILE_OFFSET, B, F, F2, I, NoiseNode, scale
from .stamped import FX_VALUES, repeat_tiles, tile_counts

ANGLE = {"angle": Param(F, 0.0, "Angle", 0.0, 1.0), "angle_random": Param(F, 0.0, "Angle random", 0.0, 1.0)}


def directional(hi: int) -> dict:
    """The parameters of the directional noises, Scale 1 up to `hi`."""
    return {**scale(1, 1, hi), **DISORDER, **ANISO, **ANGLE, **TILE_OFFSET, **NSE}


class DirectionalNoise1(NoiseNode):
    """Fine fibrous grain over soft streaked blotches: octaves of gradient noise from 2 x Scale cells across the tile,
    each averaged along the direction over a streak of a few of its cells (short for the coarse octaves, longer for
    the fine grain). Angle turns the streaks (0: horizontal), Angle random turns each octave locally and makes the
    streaks shorter and softer; Disorder moves the octaves; Tile offset shifts everything."""
    type = "directional_noise_fibrous"
    label = "Directional Noise Fibrous"
    catalog = "directional_noise_1"
    shader = "directional_fibrous"
    params = directional(8)


class DirectionalNoise(NoiseNode):
    shader = "directional"
    params = directional(8)
    scale_cells = 4


class DirectionalNoise2(DirectionalNoise):
    """Like 1, darker and with longer blotches."""
    type = "directional_noise_dark"
    label = "Directional Noise Dark"
    catalog = "directional_noise_2"
    params = directional(4)
    variant = 0


class DirectionalNoise3(DirectionalNoise):
    """Like 1, lighter and with finer blotches."""
    type = "directional_noise_light"
    label = "Directional Noise Light"
    catalog = "directional_noise_3"
    params = directional(4)
    variant = 1


class DirectionalNoise4(DirectionalNoise):
    """Smooth elongated blobs, no grain."""
    type = "directional_noise_smooth"
    label = "Directional Noise Smooth"
    catalog = "directional_noise_4"
    variant = 2


class DirectionalScratches(Node):
    """A random scattering of scratches, drawn as Cones overlaid with Max on black by the stamp engine: (32 x Scale)^2
    of them, one at a random point of each cell of a 32 x Scale grid (a stratified scattering); a Cone Pattern size.y
    cells long and .x wide (each shrunk by up to Pattern size random of itself), turned by Angle +- Angle random / 2
    (turns), its luminosity random 0-1 where a random number falls under Pattern amount (else none). Disorder moves
    each scratch around its own circle (LOOP x Disorder speed x 0.3-1 cells, a full turn looping), flattened by the
    anisotropy across its angle."""
    type = "directional_scratches"
    label = "Directional scratches"
    catalog = "directional_scratches"
    category = CATEGORY
    outputs = GRAY_OUT
    params = {
        **scale(4, 1, 16), **DISORDER, **ANISO, **ANGLE,
        "pattern_amount": Param(F, 1.0, "Pattern amount", 0.0, 1.0),
        "pattern_size": Param(F2, [0.5, 20.0], "Pattern size", 0.0, 40.0),
        "pattern_size_random": Param(F2, [0.0, 0.0], "Pattern size random", 0.0, 1.0),
        **TILE_OFFSET, **NSE,
    }
    CELLS = 32   # scratches per side of the tile at Scale 1
    LOOP = 2.5   # the Disorder circle's largest radius, in cells (fitted on the output's motion)

    def rows(self, v: dict, size: tuple[int, int], seed: int) -> np.ndarray:
        kx, ky = tile_counts(v, size)
        rng = np.random.default_rng([seed & 0xFFFFFFFF, 7])
        n = max(v["scale"], 1) * self.CELLS  # cells per side of the tile
        count = n * n
        gy, gx = np.mgrid[0:n, 0:n]
        pos = np.stack([gx.ravel(), gy.ravel()], 1) + rng.random((count, 2))
        a = min(max(v["disorder_anisotropy"], 0.0), 1.0)
        r = self.LOOP * v["disorder_speed"] * (0.3 + 0.7 * rng.random(count))
        th = 2 * math.pi * (rng.random(count) + v["disorder"])
        dx, dy = r * np.cos(th), r * np.sin(th) * (1 - a)
        t = 2 * math.pi * v["disorder_anisotropy_angle"]
        pos += np.stack([dx * math.cos(t) - dy * math.sin(t), dx * math.sin(t) + dy * math.cos(t)], 1)
        (wx, ly), (rwx, rly) = v["pattern_size"], v["pattern_size_random"]
        rows = np.zeros((count, ROW), np.float32)
        rows[:, 0] = (pos[:, 0] / n + v["tile_offset"][0]) % 1.0
        rows[:, 1] = (pos[:, 1] / n + v["tile_offset"][1]) % 1.0
        rows[:, 2] = ly * (1 - rly * rng.random(count)) / n
        rows[:, 3] = wx * (1 - rwx * rng.random(count)) / n
        rows[:, 4] = v["angle"] + (rng.random(count) - 0.5) * v["angle_random"]
        rows[:, 6] = PATTERNS.index("Cone")
        rows[:, 8:11] = np.where(rng.random(count) < v["pattern_amount"], rng.random(count), 0.0)[:, None]
        rows[:, 11] = 1.0
        rows[:, 14] = BLENDING.index("Max")
        return repeat_tiles(rows, kx, ky)

    def encode(self, ctx, job):
        draw_patterns(ctx, job, self.rows(job.values, job.size, int(job.seed)),
                      {**FX_VALUES, "background": [0.0, 0.0, 0.0, 1.0]})


class AnisotropicNoise(NoiseNode):
    """Stack of random strips fading into one another along their length."""
    type = "anisotropic_noise"
    label = "Anisotropic noise"
    catalog = "anisotropic_noise"
    params = {
        "x_amount": Param(I, 4, "X amount", 1, 512),
        "y_amount": Param(I, 256, "Y amount", 1, 512),
        "y_amount_by_resolution": Param(B, False, "Y amount by resolution"),
        "rotate": Param(B, False, "Rotate"),
        "smoothness": Param(F, 1.0, "Smoothness", 0.0, 1.0),
        "smoothness_interpolation": Param(F, 0.5, "Smoothness interpolation", 0.0, 1.0),
        **DISORDER, **NSE,
    }
