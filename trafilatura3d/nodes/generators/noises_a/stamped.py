"""The noises made of patterns: sums of the built-in patterns (nodes/shapes.wgsl) stamped at many scales. The
stamps are generated directly (no FX-Map graph to walk) and drawn by the FX-Map engine (atomic/stamp_map.py `draw_patterns`). A noise is up to three layers, each a
sum of stamps around 0.5: per level as many stamps as the level has cells, at random positions, their Color /
Luminosity random around 0.5 so that with Add Sub they lighten or darken, the spread falling by the roughness per
level. A last pass (stamped_mix.wgsl) adds the layers, a layer either as it is or folded around 0.5 (dark creases
where it crosses it), the pixel level (one stamp per pixel: hashed in the pass, not stamped) and a vertical ramp,
then offset and gain. The recipes are ours, fitted (energy per scale
band, mean, spread, over several seeds).

Scale (common.py): how many times the whole pattern repeats, a layer's first level having `cells` x
Scale cells across the tile; the levels stay the same at any resolution, as an FX-Map's, but those whose cells would be smaller than a pixel are not drawn (invisible, and with a high Scale
millions of stamps: Clouds 2 at Scale 6 would stamp 2.4 million at any size), nor those over MAX_STAMPS (at 4096
the levels of a few pixels: the pixel level `grain` stands for the finest). Disorder moves each stamp around a
loop of a quarter of its cell, along its own direction (anisotropy flattens the loop); Tile offset shifts the whole
pattern; Non-square expansion keeps the cells square.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from trafilatura3d.fxmap import BLENDING, PATTERNS
from trafilatura3d.nodes.atomic.stamp_map import ROW, FxMap, draw_patterns
from trafilatura3d.nodes.base import Input, Node, Param, ParamKind, PortType

from .common import CATEGORY, CLASSIC, GRAY_OUT, HERE, scale

# The FX-Map node's values the layers are drawn with: the stamps added over a 0.5 gray, not clamped, Roughness off
# (the layers weigh their levels themselves).
FX_VALUES = {**{k: p.default for k, p in FxMap.params.items()}, "roughness": 0.0,
             "background": [0.5, 0.5, 0.5, 1.0]}
ADD_SUB = BLENDING.index("Add Sub")
MAX_LAYERS = 3
MAX_STAMPS = 1 << 20  # stamps of one level


@dataclass(frozen=True)
class Layer:
    pattern: str        # the stamped pattern (fxmap.PATTERNS)
    levels: int         # levels of stamps, each with twice the cells of the previous
    cells: int          # cells of the first level per unit of Scale
    size: float         # stamp size in cells
    spread: float       # Color / Luminosity spread around 0.5 on the first level
    roughness: float    # spread ratio between levels
    weight: float = 1.0
    fold: float = 0.0   # > 0: the layer folded around 0.5, |value - 0.5| / sigma to this power
    sigma: float = 1.0  # the layer's typical distance from 0.5, normalizing the fold


@dataclass(frozen=True)
class Recipe:
    layers: tuple
    offset: float  # the result: offset + gain x the sum, in 0-1
    gain: float = 1.0
    grain: float = 0.0  # the pixel level: its luminosity spread
    ramp: float = 0.0   # a vertical ramp added, 0 at the middle, + ramp / 2 at the bottom (the noise does not tile)


def tile_counts(v: dict, size: tuple[int, int]) -> tuple[int, int]:
    """Non-square expansion: how many times (kx, ky) the square tile repeats across the image."""
    w, h = size
    if not v.get("non_square_expansion", True):
        return 1, 1
    return max(1, round(w / min(w, h))), max(1, round(h / min(w, h)))


def repeat_tiles(rows: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """FX-Map rows placed in the square tile (0-1), repeated kx by ky times across the image, tile by tile."""
    out = np.repeat(rows[None], kx * ky, 0)
    k = np.arange(kx * ky)
    out[:, :, 0] = (rows[None, :, 0] + (k % kx)[:, None]) / kx
    out[:, :, 1] = (rows[None, :, 1] + (k // kx)[:, None]) / ky
    out[:, :, 2] /= kx
    out[:, :, 3] /= ky
    return out.reshape(-1, ROW)


def layer_rows(layer: Layer, salt: int, index: int, v: dict, size: tuple[int, int], seed: int) -> np.ndarray:
    """A layer's stamps as FX-Map rows (atomic/stamp_map.py `rows_of`), drawn over the 0.5 gray (FX_VALUES); `salt`
    and `index` make its random stream (the node's `stream_salt`, the layer). `v`: Scale, Disorder... (missing: defaults)."""
    w, h = size
    kx, ky = tile_counts(v, size)
    rng = np.random.default_rng([seed & 0xFFFFFFFF, salt, index])
    out = [np.zeros((0, ROW), np.float32)]
    dis, speed = v.get("disorder", 0.0), v.get("disorder_speed", 1.0)
    a = min(max(v.get("disorder_anisotropy", 0.0), 0.0), 1.0)
    angle = v.get("disorder_anisotropy_angle", 0.0)
    tx, ty = v.get("tile_offset", (0.0, 0.0))
    for d in range(layer.levels):
        n = max(v.get("scale", 1), 1) * layer.cells * (1 << d)  # cells per side of the tile
        if n * kx > w or n * ky > h or n * n * kx * ky > MAX_STAMPS:  # under a pixel, or too many: not drawn
            break
        count = n * n
        pos = rng.random((count, 2))
        spread = layer.spread * layer.roughness ** d
        lums = 0.5 + (rng.random(count) - 0.5) * spread
        hx, hy = rng.random(count), rng.random(count)
        # Disorder: a loop of a quarter cell along the stamp's direction (anisotropy flattens it)
        ph = 2 * math.pi * (dis * speed + hx)
        turn = 2 * math.pi * angle + (1 - a) * 2 * math.pi * hy
        lx, ly = np.cos(ph), (1 - a) * np.sin(ph)
        px_ = (pos[:, 0] + 0.25 * (lx * np.cos(turn) - ly * np.sin(turn)) / n + tx) % 1.0  # in the tile
        py_ = (pos[:, 1] + 0.25 * (lx * np.sin(turn) + ly * np.cos(turn)) / n + ty) % 1.0
        rows = np.zeros((count, ROW), np.float32)
        rows[:, 0], rows[:, 1] = px_, py_
        rows[:, 2] = rows[:, 3] = layer.size / n
        rows[:, 6] = PATTERNS.index(layer.pattern)
        rows[:, 8:11] = (2.0 * lums - 1.0)[:, None]  # Add Sub: signed, around the 0.5 gray
        rows[:, 11] = 1.0
        rows[:, 14] = ADD_SUB
        rows[:, 15] = d
        out.append(repeat_tiles(rows, kx, ky))
    return np.concatenate(out)


def random_shards(v: dict, size: tuple[int, int], rng, count: int, top: float, octaves: float, aspect: tuple,
                  tilt: float, cut: float, slope: float, gain: float) -> np.ndarray:
    """Shards (ours, Crystal 2): `count` x Scale^2 Half Bells at uniform random positions (a sparse convolution with a
    scale-invariant density): heights `top` / Scale x 2^-(octaves x u), the number per octave growing 4x towards the
    small ones; widths `aspect` (a range) x the height; tilted up to `tilt` turns from vertical; one side cut by `cut`
    (the pattern's variation); signed amplitudes uniform in -1..1 x gain x (height / top)^slope, added with Add Sub.
    Disorder moves each shard around a loop of a tenth of its height."""
    kx, ky = tile_counts(v, size)
    s = max(v["scale"], 1)
    n = count * s * s
    if n * kx * ky > MAX_STAMPS:
        n = MAX_STAMPS // (kx * ky)
    u = np.log1p(rng.random(n) * (4.0 ** octaves - 1.0)) / (octaves * math.log(4.0))
    h = top / s * 2.0 ** (-octaves * u)
    w = h * (aspect[0] + (aspect[1] - aspect[0]) * rng.random(n))
    lum = gain * (rng.random(n) * 2.0 - 1.0) * (h * s / top) ** slope
    ang = 2 * math.pi * (rng.random(n) + v["disorder"])
    pos = rng.random((n, 2)) + 0.1 * h[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
    rows = np.zeros((n, ROW), np.float32)
    rows[:, 0:2] = pos % 1.0
    rows[:, 2] = w
    rows[:, 3] = h
    rows[:, 4] = 1.0 + tilt * (rng.random(n) * 2.0 - 1.0)
    rows[:, 5] = cut
    rows[:, 6] = PATTERNS.index("Half Bell")
    rows[:, 8:11] = lum[:, None]
    rows[:, 11] = 1.0
    rows[:, 14] = ADD_SUB
    return repeat_tiles(rows, kx, ky)


def draw_layer(ctx, job, layer: Layer, salt: int, index: int, v: dict, seed: int):
    """A layer drawn by the FX-Map engine into a new one channel 32 bit scratch (0.5 + its stamps)."""
    tex = ctx.scratch(job, True, "32f")
    draw_patterns(ctx, job, layer_rows(layer, salt, index, v, job.size, seed), FX_VALUES, dst=tex)
    return tex


class _Mix(Node):
    """The stamped noises' last pass (stamped_mix.wgsl)."""
    type = "stamped_mix"
    registered = False
    inputs = {f"layer{i}": Input(PortType.GRAY) for i in range(MAX_LAYERS)}
    outputs = GRAY_OUT
    params = {**{f"{k}{i}": Param(ParamKind.FLOAT, 0.0, "") for k in ("w", "f", "s") for i in range(MAX_LAYERS)},
              **{k: Param(ParamKind.FLOAT, 0.0, "") for k in ("offset", "gain", "grain", "ramp", "grain_px")}}

    def wgsl(self) -> str:
        return (HERE / "stamped_mix.wgsl").read_text(encoding="utf-8")


MIX = _Mix()


class StampedNoise(Node):
    """A noise drawn as FX-Map stamps (module docstring); `recipe` makes the variant."""
    category = CATEGORY
    outputs = GRAY_OUT
    params = {**scale(1, 1, 8), **CLASSIC}
    recipe: ClassVar[Recipe]

    def recipe_for(self, v: dict) -> Recipe:
        """The recipe for the node's values (a node whose parameters change the recipe overrides it)."""
        return self.recipe

    def encode(self, ctx, job):
        r, seed = self.recipe_for(job.values), int(job.seed)
        # The pixel level at the default Scale: Scale repeats the whole pattern, so below it the cells are wider
        # (in powers of two: the image still tiles).
        px = 1
        if "scale" in self.params:
            px = 1 << max(0, round(math.log2(self.params["scale"].default / max(job.values["scale"], 1))))
        values = {**{name: prm.default for name, prm in MIX.params.items()}, "offset": r.offset, "gain": r.gain,
                  "grain": r.grain, "ramp": r.ramp, "grain_px": float(px)}
        srcs = {}
        for i in range(MAX_LAYERS):
            if i < len(r.layers):
                layer = r.layers[i]
                srcs[f"layer{i}"] = (draw_layer(ctx, job, layer, self.stream_salt(), i, job.values, seed), True)
                values.update({f"w{i}": layer.weight, f"f{i}": layer.fold, f"s{i}": layer.sigma})
            else:
                srcs[f"layer{i}"] = (ctx.ev.constant((0.5, 0.5, 0.5, 1.0)), False)
        ctx.dispatch(MIX, job, values, srcs=srcs)


class Clouds2(StampedNoise):
    """Soft bright puffy clouds: one layer of Bell stamps three cells wide over six levels."""
    type = "clouds_soft"
    salt = 795  # its random stream before the rename of 15 October 2026: the same look
    label = "Clouds Soft"
    catalog = "clouds_2"
    recipe = Recipe((Layer("Bell", levels=6, cells=8, size=3.0, spread=0.14, roughness=0.7),), offset=0.74)


class Clouds1(StampedNoise):
    """Light clouds with fine dark creases: soft Bell tone, and finer Bell stamps folded around 0.5 (the creases
    where they cross it)."""
    type = "clouds_creased"
    salt = 794  # its random stream before the rename of 15 October 2026: the same look
    label = "Clouds Creased"
    catalog = "clouds_1"
    recipe = Recipe((Layer("Bell", levels=3, cells=8, size=2.5, spread=0.3, roughness=0.7),
                     Layer("Bell", levels=4, cells=32, size=2.5, spread=0.3, roughness=0.8, weight=0.4, fold=0.5,
                           sigma=0.1)), offset=0.447, gain=0.432)


class Clouds3(StampedNoise):
    """Darker billowy clouds with bright puffs: Bell stamps three cells wide over five levels."""
    type = "clouds_billowy"
    salt = 796  # its random stream before the rename of 15 October 2026: the same look
    label = "Clouds Billowy"
    catalog = "clouds_3"
    recipe = Recipe((Layer("Bell", levels=5, cells=12, size=3.0, spread=0.3, roughness=0.85),), offset=0.42,
                    gain=0.454)


class BnwSpots1(StampedNoise):
    """Rough gray spots down to the pixel: Bell stamps over seven levels and the pixel level. Roughness is the
    amplitude ratio between levels, as the FX-Map Roughness weighting (1: all with the same amplitude), the result kept
    at the default's contrast."""
    type = "bnw_noise_rough"
    salt = 1135  # its random stream before the rename of 15 October 2026: the same look
    label = "BnW Noise Rough"
    catalog = "bnw_spots_1"
    params = {**scale(1, 1, 8), **CLASSIC, "roughness": Param(ParamKind.FLOAT, 1.0, "Roughness", 0.0, 2.0)}
    LEVELS, GRAIN = 7, 0.6
    recipe = Recipe((Layer("Bell", levels=LEVELS, cells=4, size=3.0, spread=0.3, roughness=1.0),), offset=0.369,
                    gain=0.336, grain=GRAIN)

    def recipe_for(self, v: dict) -> Recipe:
        q = min(max(v["roughness"], 0.0), 2.0)
        amps = np.array([0.3 * q ** d for d in range(self.LEVELS)] + [self.GRAIN * q ** self.LEVELS])
        norm = np.sqrt((np.array([0.3] * self.LEVELS + [self.GRAIN]) ** 2).sum() / max((amps ** 2).sum(), 1e-12))
        r = self.recipe
        return Recipe((Layer("Bell", levels=self.LEVELS, cells=4, size=3.0, spread=0.3, roughness=q),), r.offset,
                      r.gain * norm, grain=self.GRAIN * q ** self.LEVELS)


class BnwSpots2(StampedNoise):
    """Rough gray spots: Bell stamps of equal amplitude over seven levels, down to a few pixels at 256."""
    type = "bnw_noise_even"
    salt = 1136  # its random stream before the rename of 15 October 2026: the same look
    label = "BnW Noise Even"
    catalog = "bnw_spots_2"
    recipe = Recipe((Layer("Bell", levels=7, cells=3, size=3.0, spread=0.3, roughness=1.0),), offset=0.458,
                    gain=0.32)


class BnwSpots3(StampedNoise):
    """Rough gray spots, smoother and lighter than BnW spots 2: smaller Bell stamps, six levels."""
    type = "bnw_noise_soft"
    salt = 1137  # its random stream before the rename of 15 October 2026: the same look
    label = "BnW Noise Soft"
    catalog = "bnw_spots_3"
    recipe = Recipe((Layer("Bell", levels=6, cells=3, size=2.0, spread=0.3, roughness=0.9),), offset=0.497,
                    gain=0.57)
