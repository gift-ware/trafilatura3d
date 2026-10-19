"""The Legacy Clouds, BnW noises, Dirt and Fractal sum generators "(Legacy)" (the documentation doesn't list them).

Ours, each fitted on the spectrum, histogram and look of the legacy node's outputs:
- Clouds Creased, Clouds Billowy, BnW Noise Rough and Soft, Dirt Patchy, Fine and gradient, Fractal Sum Gritty: a
  stack of sparse convolution layers (`Layer`, `clouds_stamps.wgsl`): random stamps (strokes, discs, bells,
  paraboloids, pyramids) on a lattice per layer, coarse layers drawn at a fraction of the output size, the stack's
  weights, lift and gamma fitted per node; Dirt Sparse is Dirt Patchy through a Levels.
- Clouds Soft and BnW Noise Even: the Moisture noise shader with their own fit; Fractal Sum Blobby: a blurred white
  noise; Dirt Coarse: a white noise of one value per pixel smeared by a small kernel, longer along x
  (`clouds_dirt_coarse.wgsl`)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from trafilatura3d.nodes.base import Input, Node, Output, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import blur2d

from ._lib import DISORDER, LegacyNode, apply, levels, mix, sized, target, white
from ..noises_b.noises import MOISTURE_FIT, MoistureNoise1

MOISTURE = MoistureNoise1()


# --- the stamps noise (ours) -----------------------------------------------------------------------------------------

PYRAMID, DISC, BELL, PARABOLOID = range(4)


@dataclass(frozen=True)
class Layer:
    """One layer of the stamps noise (clouds_stamps.wgsl): `cells` x `cells` lattice cells across the tile, `count`
    slots per cell holding a stamp with probability `density`, stamps `length` cells long (at most 3) and `aspect`
    times that wide, turned at random (`turn`), heights uniform in `h_low` .. `h_high`; drawn at 1 / `div` of the
    output size and added with `weight` (see the shader for the rest)."""
    cells: int
    weight: float = 1.0
    count: int = 1
    div: int = 1
    length: float = 1.0
    aspect: float = 1.0
    turn: bool = False
    shape: int = PYRAMID
    density: float = 1.0
    patchiness: float = 0.0
    patch_cells: int = 8
    h_low: float = 0.0
    h_high: float = 1.0
    width_var: float = 0.0
    clamp_sum: bool = False
    grain: float = 0.0
    grain_cells: int = 0
    ramp: tuple = ()  # the vertical density profile (top, a third, two thirds, bottom); () = none


class _Stamps(Node):
    """One layer of the stamps noise over the coarser ones (clouds_stamps.wgsl)."""
    type = "clouds_stamps"
    registered = False
    inputs = {"under": Input(PortType.GRAY)}
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {**{k: ParamKind.INT for k in ("cells", "count", "turn", "shape", "patch_cells", "clamp_sum",
                                                       "grain_cells", "ramp", "last", "salt")},
                         **{k: ParamKind.FLOAT for k in ("weight", "length", "aspect", "density", "patchiness",
                                                         "h_low", "h_high", "width_var", "grain", "r0", "r1", "r2",
                                                         "r3", "disorder", "lift", "gamma")}}

    def wgsl(self) -> str:
        return (Path(__file__).parent / "clouds_stamps.wgsl").read_text(encoding="utf-8")


STAMPS = _Stamps()


def stamp_layers(ctx, job, seed: float, layers, lift: float, gamma: float, disorder: float, dst=None):
    """The layers drawn at `seed` coarse to fine, each over the previous one; the last at the job's size into
    `target(dst)`, lift + the sum clamped through `gamma`. `disorder` moves the stamps around their cells (1: a whole
    cell)."""
    under, seeded = None, dataclasses.replace(job, seed=seed)
    for k, layer in enumerate(layers):
        last = k == len(layers) - 1
        j = sized(seeded, (max(1, job.size[0] // layer.div), max(1, job.size[1] // layer.div)))
        out = target(ctx, job, dst) if last else ctx.scratch(j, True, "32f")
        values = {f.name: getattr(layer, f.name) for f in dataclasses.fields(layer) if f.name not in ("div", "ramp")}
        values.update({f"r{i}": r for i, r in enumerate(layer.ramp or (1.0,) * 4)})
        values.update(turn=int(layer.turn), clamp_sum=int(layer.clamp_sum), ramp=int(bool(layer.ramp)),
                      last=int(last), salt=k, disorder=disorder, lift=lift, gamma=gamma)
        ctx.dispatch(STAMPS, j, values, srcs={"under": under} if under else {}, dsts={"output": out})
        under = (out, True)
    return under


class StampNoise(LegacyNode):
    """A legacy noise made with the stamps noise: LAYERS, then LIFT + the sum through GAMMA; Disorder (when the node
    has it) moves the stamps around their cells."""
    LAYERS: ClassVar[tuple] = ()
    LIFT: ClassVar[float] = 0.0
    GAMMA: ClassVar[float] = 1.0

    def image(self, ctx, job, v, seed, dst=None):
        return stamp_layers(ctx, job, seed, self.LAYERS, self.LIFT, self.GAMMA, 0.01 * v.get("disorder", 0.0), dst)


# --- Clouds ----------------------------------------------------------------------------------------------------------

def _strokes(cells: int, weight: float, div: int, length: float = 2.0) -> Layer:
    return Layer(cells, weight, count=6, div=div, length=length, aspect=0.2, turn=True)


class Clouds1Legacy(LegacyNode):
    """Light clouds with dark creases (ours): the Moisture noise shader (noises_b/moisture_noise.wgsl) with its own fit
    (FIT: octaves from SCALE x 4 cells across, the detail growing where the coarser octaves are strong: clustered dark
    patches), times a fibrous texture of thin dark strokes turned at random (FIBERS, the stamps noise); fitted on the
    spectrum, histogram and look of the legacy node's outputs. Disorder moves the clouds and the strokes."""
    type = "clouds_creased_legacy"
    label = "Clouds Creased (Legacy)"
    params = DISORDER
    FIT = {**MOISTURE_FIT, "mo_rough": 0.72, "mo_first": 0.6, "mo_octaves": 5, "mo_gain": 0.39, "mo_gain_up": 0.0,
           "mo_gain_up2": 0.0, "mo_area": 0.0, "mo_cluster_base": 0.6, "mo_cluster": 0.9, "mo_lift": 0.2}
    SCALE = 4
    FIBERS = (_strokes(20, -0.08, 4), _strokes(40, -0.12, 2), _strokes(80, -0.08, 1))  # the last at full size

    def image(self, ctx, job, v, seed, dst=None):
        values = {**{k: prm.default for k, prm in MOISTURE.params.items()}, **self.FIT, "scale": self.SCALE,
                  "disorder": 0.01 * v["disorder"], "mo_seed": float(int(seed) % 9973) * 0.731}
        half = sized(job, (max(1, job.size[0] // 2), max(1, job.size[1] // 2)))  # smooth: drawn at half size
        clouds = ctx.scratch(half, True, "32f")
        ctx.dispatch(MOISTURE, half, values, dsts={"output": clouds})
        fibers = stamp_layers(ctx, job, seed + 7.0, self.FIBERS, 1.0, 1.0, 0.01 * v["disorder"])
        return mix(ctx, job, fibers, (clouds, True), "Multiply", dst=dst)


class Clouds2Legacy(LegacyNode):
    """Soft bright clouds: the Moisture noise shader (noises_b/moisture_noise.wgsl, ours) with its own fit (FIT: 8
    octaves from 4 cells across, the first one weak, drawn inverted: negative gain), then 1 - x^2.5 (Levels mid
    GAMMA_MID, output inverted): the clouds lifted, a soft dark tail; fitted on the spectrum and histogram of the
    legacy node's outputs."""
    type = "clouds_soft_legacy"
    label = "Clouds Soft (Legacy)"
    params = {}
    FIT = {**MOISTURE_FIT, "mo_rough": 0.6, "mo_first": 0.32, "mo_octaves": 8, "mo_gain": -0.28, "mo_gain_up": 0.0,
           "mo_gain_up2": 0.0, "mo_area": 0.0, "mo_cluster_base": 1.0, "mo_cluster": 0.0, "mo_lift": 0.115}
    GAMMA_MID = 0.59375

    def image(self, ctx, job, v, seed, dst=None):
        values = {**{k: prm.default for k, prm in MOISTURE.params.items()}, **self.FIT,
                  "mo_seed": float(int(seed) % 9973) * 0.731}
        noise = ctx.scratch(job, True, "32f")
        ctx.dispatch(MOISTURE, job, values, dsts={"output": noise})
        return levels(ctx, job, (noise, True), mid=self.GAMMA_MID, out_low=1.0, out_high=0.0, dst=dst)


def _billows(cells: int, weight: float, div: int) -> Layer:
    return Layer(cells, weight, div=div, length=3.0, aspect=0.67, turn=True, h_low=-1.0)


class Clouds3Legacy(StampNoise):
    """Darker billowy clouds (ours): signed pyramids 3 x 2 cells turned at random, one per cell, 16 to 128 cells across
    (drawn at a quarter and half the output size), around a gray lift, plus a faint grain of 2 pixels at 1024; fitted
    on the spectrum, histogram and look of the legacy node's outputs."""
    type = "clouds_billowy_legacy"
    label = "Clouds Billowy (Legacy)"
    params = {}
    LAYERS = (_billows(16, 0.08338, 4), _billows(32, 0.156, 4), _billows(64, 0.0936, 2), _billows(128, 0.0616, 2),
              Layer(1, 0.0, count=0, grain=0.02659, grain_cells=512))
    LIFT, GAMMA = 0.5015, 1.276


# --- BnW noises ------------------------------------------------------------------------------------------------------

def _spots(cells: int, weight: float, div: int, length: float = 1.5, aspect: float = 0.67, h_low: float = 0.0
           ) -> Layer:
    return Layer(cells, weight, div=div, length=length, aspect=aspect, turn=True, shape=PARABOLOID, h_low=h_low)


class BnwSpots1Legacy(StampNoise):
    """Rough white spots at every scale (ours): paraboloids 1.5 x 1 cells turned at random, one per cell, 2 to 256 cells
    across (the coarse ones drawn small: soft), added on black; fitted on the spectrum, histogram and look of the
    legacy node's outputs."""
    type = "bnw_noise_rough_legacy"
    label = "BnW Noise Rough (Legacy)"
    params = DISORDER
    LAYERS = (_spots(2, 0.06689, 16), _spots(4, 0.2533, 16), _spots(8, 0.1793, 16), _spots(16, 0.1395, 4),
              _spots(32, 0.2418, 4), _spots(64, 0.1692, 4), _spots(128, 0.2213, 2), _spots(256, 0.1626, 1))
    LIFT, GAMMA = -0.02533, 0.9418


def _soft(cells: int, weight: float, div: int) -> Layer:
    return _spots(cells, weight, div, 2.0, 0.75, -1.0)


class BnwSpots3Legacy(StampNoise):
    """Soft gray spots (ours): signed paraboloids 2 x 1.5 cells turned at random, one per cell, 2 to 64 cells across
    drawn small (very soft), and finer ones 1.5 x 1 cells, 128 to 512 across (the grain), around a mid gray; fitted on
    the spectrum, histogram and look of the legacy node's outputs."""
    type = "bnw_noise_soft_legacy"
    label = "BnW Noise Soft (Legacy)"
    params = DISORDER
    LAYERS = (_soft(2, 0.05768, 32), _soft(4, 0.0959, 32), _soft(8, 0.08704, 32), _soft(16, 0.1073, 4),
              _soft(32, 0.02073, 4), _soft(64, 0.1289, 4), _spots(128, -0.01689, 2, h_low=-1.0),
              _spots(256, 0.1311, 1, h_low=-1.0), _spots(512, 0.05903, 1, h_low=-1.0))
    LIFT, GAMMA = 0.514, 1.132


class _Specks(Node):
    """Sparse dots over an image (specks.wgsl)."""
    type = "legacy_specks"
    registered = False
    inputs = {"input": Input(PortType.GRAY)}
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"cells": ParamKind.INT, "density": ParamKind.FLOAT, "radius": ParamKind.FLOAT,
                         "bright": ParamKind.FLOAT, "amount": ParamKind.FLOAT}

    def wgsl(self) -> str:
        return (Path(__file__).parent / "specks.wgsl").read_text(encoding="utf-8")


SPECKS = _Specks()


class BnwSpots2Legacy(LegacyNode):
    """Even gray blotches at every scale with a fine grain: the Moisture noise shader (noises_b/moisture_noise.wgsl,
    ours) with its own fit (FIT: 9 octaves from 4 cells across, each finer one a little stronger, the first one
    strong), plus a white noise grain (Add Sub GRAIN) and sparse bright and dark specks (specks.wgsl, SPECKS); fitted
    on the spectrum, histogram and speck counts of the legacy node's outputs. Disorder moves the blotches along a
    loop."""
    type = "bnw_noise_even_legacy"
    label = "BnW Noise Even (Legacy)"
    params = DISORDER
    FIT = {**MOISTURE_FIT, "mo_rough": 1.03, "mo_first": 1.8, "mo_octaves": 9, "mo_gain": 0.1, "mo_gain_up": 0.0,
           "mo_gain_up2": 0.0, "mo_area": 0.0, "mo_cluster_base": 1.0, "mo_cluster": 0.0, "mo_lift": -0.06}
    GRAIN = 0.04
    SPECKS = {"cells": 256, "density": 0.55, "radius": 0.45, "bright": 0.68, "amount": 0.34}

    def image(self, ctx, job, v, seed, dst=None):
        values = {**{k: prm.default for k, prm in MOISTURE.params.items()}, **self.FIT,
                  "disorder": 0.01 * v["disorder"], "mo_seed": float(int(seed) % 9973) * 0.731}
        noise = ctx.scratch(job, True, "32f")
        ctx.dispatch(MOISTURE, job, values, dsts={"output": noise})
        grained = mix(ctx, job, (noise, True), white(ctx, job, job.size), "Add sub", self.GRAIN)
        return apply(ctx, job, SPECKS, {"input": grained}, dst, **self.SPECKS)


# --- Dirt ------------------------------------------------------------------------------------------------------------

def _discs(cells: int, weight: float, density: float, h_low: float, h_high: float, patchiness: float) -> Layer:
    return Layer(cells, weight, count=3, shape=DISC, density=density, patchiness=patchiness, patch_cells=12,
                 h_low=h_low, h_high=h_high, width_var=0.5, clamp_sum=True)


class Dirt1Legacy(StampNoise):
    """Fine white specks in patches (ours): small discs (one cell, narrowed at random), up to three per cell, 128 and
    256 cells across, their density times a smooth random field (dense and empty patches), each layer's sum clamped,
    on black; fitted on the spectrum, histogram and look of the legacy node's outputs."""
    type = "dirt_patchy_legacy"
    label = "Dirt Patchy (Legacy)"
    params = DISORDER
    LAYERS = (_discs(128, 1.0, 0.3606, 0.1172, 0.1735, 3.078), _discs(256, 1.05, 0.5048, 0.1172, 0.1178, 1.01))


class Dirt2Legacy(LegacyNode):
    """Sparser white specks (ours): Dirt Patchy (at seed + 6) through Levels in low LOW, fitted on the legacy node's
    outputs."""
    type = "dirt_sparse_legacy"
    label = "Dirt Sparse (Legacy)"
    params = DISORDER
    LOW = 0.385

    def image(self, ctx, job, v, seed, dst=None):
        return levels(ctx, job, DIRT_1.instance(ctx, job, {"disorder": v["disorder"]}, seed + 6), low=self.LOW, dst=dst)


class Dirt3Legacy(StampNoise):
    """Film dust (ours): small square pyramids (one cell), up to three per cell, 128 and 256 cells across, each layer's
    sum clamped, on black; fitted on the spectrum, histogram and look of the legacy node's outputs."""
    type = "dirt_fine_legacy"
    label = "Dirt Fine (Legacy)"
    params = DISORDER
    LAYERS = (Layer(128, count=3, density=0.2902, h_high=0.5708, clamp_sum=True),
              Layer(256, 0.5175, count=3, density=0.2371, h_high=0.5863, clamp_sum=True))


class _DirtCoarse(Node):
    """Dirt Coarse's pass (clouds_dirt_coarse.wgsl)."""
    type = "clouds_dirt_coarse"
    registered = False
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {k: ParamKind.FLOAT for k in ("kx1", "kx2", "ky1", "ky2", "low", "high", "gamma")}

    def wgsl(self) -> str:
        return (Path(__file__).parent / "clouds_dirt_coarse.wgsl").read_text(encoding="utf-8")


DIRT_COARSE = _DirtCoarse()


class Dirt4Legacy(LegacyNode):
    """Short horizontal dashes in a fine grain (ours, clouds_dirt_coarse.wgsl): a white noise of one value per pixel
    smeared by a small separable kernel, longer along x, then a Levels. The pixel stays the grain at every resolution
    and the kernel grows a little with it (KERNEL: each weight a + b log2(size / 256) per axis), so finer outputs are
    smoother and less contrasted; fitted on the mean, spread, histogram and pixel correlations of the legacy node's
    outputs from 256 to 2048."""
    type = "dirt_coarse_legacy"
    label = "Dirt Coarse (Legacy)"
    params = {}
    KERNEL = {"kx1": (0.2049, 0.1004), "kx2": (0.0488, 0.0023), "ky1": (0.0599, 0.0294), "ky2": (0.041, 0.0014)}
    LEVELS = {"low": -0.0505, "high": 1.012, "gamma": 2.658}

    def image(self, ctx, job, v, seed, dst=None):
        out = target(ctx, job, dst)
        tx, ty = (math.log2(max(s, 1) / 256.0) for s in job.size)
        kernel = {k: a + b * (tx if k.startswith("kx") else ty) for k, (a, b) in self.KERNEL.items()}
        ctx.dispatch(DIRT_COARSE, dataclasses.replace(job, seed=float(seed)), {**kernel, **self.LEVELS},
                     dsts={"output": out})
        return out, True


RISE = (0.07595, 0.3155, 0.5645, 0.5703)  # Dirt gradient's density from the top to the bottom


def _bells(cells: int, weight: float, div: int, count: int = 3, length: float = 3.0, ramp: tuple = RISE) -> Layer:
    return Layer(cells, weight, count=count, div=div, length=length, shape=BELL, ramp=ramp)


class DirtGradientLegacy(StampNoise):
    """Dirt rising from the bottom (ours): bells 3 cells wide, two or three per cell, 2 to 512 cells across (the coarse
    ones drawn small), their density rising from the top to the bottom (RISE; the two coarsest everywhere); fitted on
    the spectrum, histogram, row profile and look of the legacy node's outputs. Vertically it does not tile."""
    type = "dirt_gradient_legacy"
    label = "Dirt gradient (Legacy)"
    seamless = False
    params = {}
    LAYERS = (_bells(2, 0.0124, 32, ramp=()), _bells(4, -0.03289, 32, ramp=()), _bells(8, 0.0728, 16),
              _bells(16, 0.04191, 8), _bells(32, 0.04993, 4), _bells(64, 0.0144, 2), _bells(128, 0.01632, 1),
              _bells(256, 0.06191, 1, 2), _bells(512, 0.02142, 1, 2, 2.0))
    LIFT, GAMMA = 0.01887, 1.218


# --- Fractal sum -----------------------------------------------------------------------------------------------------

def _gritty(cells: int, weight: float, div: int, grain: float = 0.0) -> Layer:
    return Layer(cells, weight, div=div, length=2.0, shape=BELL, h_low=-1.0, grain=grain)


class FractalSum2Legacy(StampNoise):
    """Gritty fractal noise (ours): signed bells 2 cells wide, one per cell, 2 to 512 cells across (the coarse ones
    drawn small), plus a white noise of one value per pixel, around a gray lift; fitted on the spectrum, histogram and
    look of the legacy node's outputs."""
    type = "fractal_sum_gritty_legacy"
    label = "Fractal Sum Gritty (Legacy)"
    params = {}
    LAYERS = (_gritty(2, -0.08636, 16), _gritty(4, 0.1121, 16), _gritty(8, 0.1285, 8), _gritty(16, 0.1715, 8),
              _gritty(32, 0.05975, 4), _gritty(64, 0.1638, 2), _gritty(128, 0.2302, 1), _gritty(256, 0.01637, 1),
              _gritty(512, 0.2055, 1, 0.2573))
    LIFT, GAMMA = 0.524, 1.755


class FractalSum4Legacy(LegacyNode):
    """Interlaced bright and dark blobs (ours): a gaussian random field, a white noise under a gaussian blur of sigma
    SIGMA (pixels at 256: the field's grain), plus (Add Sub GRAIN) a white noise drawn at 1 / GRAIN_DIV of the size and
    enlarged (a soft grain), stretched by Levels (LEVELS: in low / high, shifted to the bright side); fitted on the
    spectrum and histogram of the legacy node's outputs."""
    type = "fractal_sum_blobby_legacy"
    label = "Fractal Sum Blobby (Legacy)"
    params = {}
    SIGMA, GRAIN, GRAIN_DIV = 1.6, 0.006, 4
    LEVELS = {"low": 0.457, "high": 0.526}

    def image(self, ctx, job, v, seed, dst=None):
        field = (blur2d(ctx, job, white(ctx, job, job.size), True, self.SIGMA * job.size[0] / 256.0), True)
        if self.GRAIN:
            small = (max(1, job.size[0] // self.GRAIN_DIV), max(1, job.size[1] // self.GRAIN_DIV))
            grain = white(ctx, dataclasses.replace(job, seed=job.seed + 11.0), small)
            field = mix(ctx, job, field, grain, "Add sub", self.GRAIN)
        return levels(ctx, job, field, dst=dst, **self.LEVELS)


DIRT_1 = Dirt1Legacy()
