"""The Legacy fur, grass, wood, scratch, directional and drip noises (Fur Straight/Wavy, Directional noise, Herb, Wood
Fibers Dark/Light, Scratches Parallel/Scattered/Curved, Fluid "(Legacy)"), the versions older materials use. Our methods,
fitted on the outputs of the earlier legacy nodes (spectrum by octave, histogram, correlations along and across,
orientations, and the look):
- streaky noises (wood fibres, directional noise): fractal streaks, a sparse convolution of stretched bells over
  octaves on anisotropic lattices (organic_streaks.wgsl);
- strands and scratches: lists of thin gaussian (or crescent) strokes from our own distributions (`strands`, at
  stratified spots: `spread`), drawn in one pass by the stamp engine (atomic/stamp_map.py `draw_patterns`);
- grass: pixel noise filtered by a two-blade kernel around a jittered point (organic_grass.wgsl);
- drips: the Fluid drips with the legacy node's units and its flow (`FluidLegacy`)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

from ._lib import (ADD_SUB, DISORDER, IDENTITY, MAX, F, LegacyNode, draw, fx_values, sized, stamps, stream, target,
                   xf)


# --- shaders and strokes ----------------------------------------------------------------------------------------------

def seed_offset(seed: float) -> float:
    """A shader seed from a node's seed (the hashes take it as a float)."""
    return float(int(seed) % 9973) * 0.731 + 0.37


class _Streaks(Node):
    """Fractal streaks (organic_streaks.wgsl)."""
    type = "organic_streaks"
    registered = False
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"st_weights": ParamKind.FLOAT4, "st_axis": ParamKind.INT, "st_octaves": ParamKind.INT,
                         "st_reach": ParamKind.INT, "st_reach_x": ParamKind.INT, "st_delay": ParamKind.INT,
                         "st_bands": ParamKind.INT,
                         **{k: ParamKind.FLOAT for k in ("st_cells_x", "st_cells_y", "st_jitter", "st_width",
                                                         "st_len_lo", "st_len_hi", "st_center", "st_rough",
                                                         "st_min_px", "st_lift", "st_gain", "st_gamma", "st_seed",
                                                         "st_shift")}}

    def wgsl(self) -> str:
        return (Path(__file__).parent / "organic_streaks.wgsl").read_text(encoding="utf-8")


STREAKS = _Streaks()


class _Grass(Node):
    """Grass blades (organic_grass.wgsl)."""
    type = "organic_grass"
    registered = False
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"gr_weights": ParamKind.FLOAT4, "gr_turns": ParamKind.FLOAT2, "gr_core": ParamKind.FLOAT,
                         "gr_length": ParamKind.INT, "gr_fade": ParamKind.FLOAT, "gr_cells": ParamKind.INT,
                         "gr_blotch": ParamKind.FLOAT, "gr_lift": ParamKind.FLOAT, "gr_gain": ParamKind.FLOAT,
                         "gr_seed": ParamKind.FLOAT, "gr_sparse": ParamKind.FLOAT, "gr_jitter": ParamKind.FLOAT,
                         "gr_jitter_px": ParamKind.FLOAT}

    def wgsl(self) -> str:
        return (Path(__file__).parent / "organic_grass.wgsl").read_text(encoding="utf-8")


GRASS = _Grass()


def spread(k: int, rng: np.random.Generator, fill: float = 1.0) -> np.ndarray:
    """k points (k, 2) over the tile, stratified: one at a random spot of k random cells of a square grid of about
    k / `fill` cells (1: every cell, an even density; less: clumps and gaps)."""
    g = max(1, math.ceil(math.sqrt(k / fill)))
    cell = rng.permutation(g * g)[:k]
    return (np.stack([cell % g, cell // g], 1) + rng.random((k, 2))) / g


def strands(k: int, rng: np.random.Generator, width: float, length: tuple, turn: np.ndarray, gain: float,
            slide: float = 0.0, pattern: str = "Gaussian", at: np.ndarray | None = None,
            blend: int = ADD_SUB) -> np.ndarray:
    """Stamp rows (`_lib.stamps`) of k strands (`pattern`) `width` wide (uv), `length` = (lo, hi) long (uniform), at
    the stratified spots of `spread` (or `at`), turned `turn` turns (0: along y), brightness `gain`, each slid along
    itself by up to `slide` of the image; added (or `blend`)."""
    size = np.stack([np.full(k, width), rng.uniform(length[0], length[1], k)], 1)
    pos = spread(k, rng) if at is None else at
    if slide:
        pos = pos + slide * rng.random(k)[:, None] * np.stack([np.sin(2 * np.pi * turn), np.cos(2 * np.pi * turn)], 1)
    return stamps(pos, size, pattern, np.full(k, float(gain)), 0, blend, rotation=turn)


def streaks(ctx, job, fit: dict, seed: float, salt: float, dst=None, shift: float = 0.0):
    """Fractal streaks with the values `fit` (organic_streaks.wgsl's, without the prefix; w0..w3 the first octaves'
    weights) at `seed` (+ the node's `salt`), each octave slid along the streaks by up to `shift` of the image."""
    out = target(ctx, job, dst)
    reach = lambda r: math.ceil(r + 0.5 * fit["jitter"] + 0.5) - 1  # noqa: E731  the cells a bell reaches
    values = {"st_center": 0.0, "st_bands": 0, **{f"st_{k}": x for k, x in fit.items() if k not in ("w0", "w1", "w2", "w3")}}
    ctx.dispatch(STREAKS, job, {**values, "st_weights": [fit[f"w{k}"] for k in range(4)],
                                "st_reach": max(1, min(4, reach(fit["len_hi"]))),
                                "st_reach_x": max(1, min(2, reach(fit["width"]))), "st_seed": seed_offset(seed) + salt,
                                "st_shift": shift},
                 dsts={"output": out})
    return out, True


# --- Fur ---------------------------------------------------------------------------------------------------------------

class Fur1Legacy(LegacyNode):
    """Upright fur strands (ours): long thin gaussian strands at even spots over the tile (`spread`), leaning
    TILT turns either way, plus four times as many at half the size and a little dimmer, added on black in one
    pass and clamped; count, width, lengths, lean and gains fitted on the spectrum, histogram, correlations and
    orientations of the legacy node's outputs. Disorder slides each strand along itself by up to Disorder / 100 of
    the image."""
    type = "fur_straight_legacy"
    label = "Fur Straight (Legacy)"
    params = DISORDER
    COUNT, WIDTH, LENGTH, TILT, GAIN = 512, 0.0156, (0.595, 0.678), (-0.0476, 0.0707), 0.465
    FINE_SCALE, FINE_GAIN = 0.504, 0.40

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        out = []
        for k, scale, gain in ((self.COUNT, 1.0, self.GAIN), (4 * self.COUNT, self.FINE_SCALE, self.FINE_GAIN)):
            out.append(strands(k, rng, scale * self.WIDTH, (scale * self.LENGTH[0], scale * self.LENGTH[1]),
                               rng.uniform(*self.TILT, k), gain, 0.01 * v["disorder"]))
        return np.concatenate(out)

    def image(self, ctx, job, v, seed, dst=None):
        return draw(ctx, job, self.rows(v, stream(seed, self)), fx_values(0.0, 1.0, 0.0), clamp=True, dst=dst)


class Fur2Legacy(LegacyNode):
    """Wavy fur (ours): gaussian strands lying along x at even spots over the tile (`spread`), plus four times as
    many smaller and dimmer ones, each turned by a flow field of two crossed sine waves at its spot: (AMP + Waves
    rotation / 2) x (cos(2 pi f y) + sin(2 pi f x)) turns, f = 2 / (Waves scale + 1) waves across the tile (it tiles
    at 0 and 1); added on black in one pass and clamped. Count, sizes and gains fitted on the spectrum, histogram,
    correlations and orientations of the legacy node's outputs; every strand moves within a square 0.01 + Disorder / 20
    of the image wide."""
    type = "fur_wavy_legacy"
    label = "Fur Wavy (Legacy)"
    params = {"disorder": Param(F, 0.0, "Disorder", 0.0, 1.0),
              "waves_scale": Param(F, 0.0, "Waves scale", 0.0, 1.0),
              "waves_rotation": Param(F, 0.0, "Waves rotation", 0.0, 1.0)}
    COUNT, WIDTH, LENGTH, GAIN, AMP = 3600, 0.0117, (0.1648, 0.3223), 0.1457, 0.054
    FINE_SCALE, FINE_GAIN = 0.4673, 0.11

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        f, amp, out = 2.0 / (v["waves_scale"] + 1.0), self.AMP + 0.5 * v["waves_rotation"], []
        for k, scale, gain in ((self.COUNT, 1.0, self.GAIN), (4 * self.COUNT, self.FINE_SCALE, self.FINE_GAIN)):
            at = spread(k, rng)
            turn = 0.25 + amp * (np.cos(2 * np.pi * f * at[:, 1]) + np.sin(2 * np.pi * f * at[:, 0]))
            out.append(strands(k, rng, scale * self.WIDTH, (scale * self.LENGTH[0], scale * self.LENGTH[1]), turn, gain,
                               at=at + (0.01 + 0.05 * v["disorder"]) * (rng.random((k, 2)) - 0.5)))
        return np.concatenate(out)

    def image(self, ctx, job, v, seed, dst=None):
        return draw(ctx, job, self.rows(v, stream(seed, self)), fx_values(0.0, 1.0, 0.0), clamp=True, dst=dst)


# --- Directional noise and wood fibres -----------------------------------------------------------------------------------

class DirectionnalNoiseLegacy(LegacyNode):
    """Horizontal streaks (ours): fractal streaks (organic_streaks.wgsl) along x, from 4 x 4 cells across the tile
    (cloudy, the first octaves' cells growing longer) to the pixel, heights partly signed; drawn at half the width (the
    streaks hold no detail finer than two pixels along x) and stretched back. Fitted on the spectrum, histogram and
    correlations along and across of the legacy node's outputs."""
    type = "directionnal_noise_legacy"
    label = "Directional noise (Legacy)"
    params = {}
    FIT = {"axis": 1, "delay": 3, "octaves": 9, "cells_x": 4.0, "cells_y": 4.0, "jitter": 1.0, "width": 0.798,
           "len_lo": 0.456, "len_hi": 1.0, "center": 0.2607, "w0": 0.382, "w1": 0.248, "w2": 1.795, "w3": 1.738,
           "rough": 0.8836, "min_px": 1.0, "lift": 0.2123, "gain": 0.157, "gamma": 1.46}

    def image(self, ctx, job, v, seed, dst=None):
        half = streaks(ctx, sized(job, (max(1, job.size[0] // 2), job.size[1])), self.FIT, seed, 0.37)
        return xf(ctx, job, half, IDENTITY, job.size, dst=dst)


class WoodFibers1Legacy(LegacyNode):
    """Dark wood fibres (ours): fractal streaks (organic_streaks.wgsl) running down the image, from 8 x 2 cells
    across the tile to the pixel, the first octave bands over the whole height (strongest), the others about as strong
    as the next, raised to a power for the dark, bright-tailed histogram; fitted on the spectrum, histogram and correlations along and across of the
    legacy node's outputs. Disorder slides each octave down by up to Disorder / 100 of the image."""
    type = "wood_fibers_dark_legacy"
    label = "Wood Fibers Dark (Legacy)"
    params = DISORDER
    FIT = {"axis": 0, "delay": 0, "bands": 1, "octaves": 8, "cells_x": 8.0, "cells_y": 2.0, "jitter": 1.0,
           "width": 0.762, "len_lo": 0.02, "len_hi": 1.944, "w0": 4.414, "w1": 2.349, "w2": 1.242, "w3": 1.653,
           "rough": 0.9758, "min_px": 1.0, "lift": 0.0331, "gain": 0.0731, "gamma": 1.499}

    def image(self, ctx, job, v, seed, dst=None):
        return streaks(ctx, job, self.FIT, seed, 0.11, dst, 0.01 * v["disorder"])


class WoodFibers2Legacy(LegacyNode):
    """Light wood grain (ours): fractal streaks (organic_streaks.wgsl) running down the image: bands over the whole
    height on 8 columns (the 4-column octave weighs 0), then streaks from 16 x 4 cells to the pixel, drawn dark on a
    light gray (negative gain) for the darker tail of the histogram;
    fitted on the spectrum, histogram and correlations along and across of the legacy node's outputs."""
    type = "wood_fibers_light_legacy"
    label = "Wood Fibers Light (Legacy)"
    params = {}
    FIT = {"axis": 0, "delay": 0, "bands": 2, "octaves": 9, "cells_x": 4.0, "cells_y": 1.0, "jitter": 1.0,
           "width": 0.776, "len_lo": 0.723, "len_hi": 2.099, "w0": 0.0, "w1": 4.155, "w2": 2.707, "w3": 2.4,
           "rough": 0.9586, "min_px": 1.0, "lift": 0.92, "gain": -0.0436, "gamma": 0.8}

    def image(self, ctx, job, v, seed, dst=None):
        return streaks(ctx, job, self.FIT, seed, 0.23, dst)


# --- Herb ----------------------------------------------------------------------------------------------------------------

class Herb1Legacy(LegacyNode):
    """Grass blades (ours, organic_grass.wgsl): pixel noise, skewed towards sparse bright dots, filtered by a kernel
    of a small round core and two short fading lines in two directions (blades), the lines' sum partly replaced by
    their larger; its taps read around a point jittered a few pixels by a smooth random field (crumpled, curly
    blades), minus soft blotches. Kernel, dots, jitter and blotches fitted on the autocorrelation (the two blade
    directions), spectrum and histogram of the legacy node's outputs. The grain is the pixel's, as the legacy node's."""
    type = "herb_legacy"
    label = "Herb (Legacy)"
    params = {}
    FIT = {"gr_weights": [1.0, 0.4505, 0.4436, 0.5807], "gr_turns": [0.0492, 0.2935], "gr_core": 1.009,
           "gr_length": 16, "gr_fade": 4.778, "gr_sparse": 3.027, "gr_jitter": 2.049, "gr_jitter_px": 6.94,
           "gr_cells": 23, "gr_blotch": 0.0985, "gr_lift": -0.006, "gr_gain": 0.28}

    def image(self, ctx, job, v, seed, dst=None):
        out = target(ctx, job, dst)
        ctx.dispatch(GRASS, job, {**self.FIT, "gr_seed": seed_offset(seed)}, dsts={"output": out})
        return out, True


# --- Scratches -----------------------------------------------------------------------------------------------------------

class Scratches1Legacy(LegacyNode):
    """Parallel scratches (ours): COUNT thin gaussian strands at even spots over the tile (`spread`, each moved
    sideways by up to SIDE of the image), WIDTH x Pattern size width / 100 wide and LENGTH x Pattern size height /
    100 long (3% longer at most), turned Rotation + up to Rotation variation degrees, added on black and clamped;
    width, length, sideways spread and gain fitted on the spectrum, histogram and orientations of the legacy node's
    outputs. Disorder slides each scratch along itself by up to Disorder / 100 of the image."""
    type = "scratches_parallel_legacy"
    label = "Scratches Parallel (Legacy)"
    params = {"pattern_size_width": Param(F, 100.0, "Pattern size width", 0.0, 1000.0),
              "pattern_size_height": Param(F, 100.0, "Pattern size height", 0.0, 1000.0),
              "rotation_var": Param(F, 35.0, "Rotation variation", 0.0, 360.0),
              "disorder": Param(F, 0.0, "Disorder", 0.0, 100.0),
              "rotation": Param(F, 15.0, "Rotation", -360.0, 360.0)}
    COUNT, WIDTH, LENGTH, GAIN, SIDE = 1024, 0.0042, 0.187, 0.52, 0.1

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        k = self.COUNT
        turn = (v["rotation"] + v["rotation_var"] * rng.random(k)) / 360.0
        length = 0.01 * self.LENGTH * v["pattern_size_height"]
        at = spread(k, rng) + [self.SIDE, 0.0] * rng.random((k, 2))
        return strands(k, rng, 0.01 * self.WIDTH * v["pattern_size_width"], (length, 1.03 * length), turn, self.GAIN,
                       0.01 * v["disorder"], at=at)

    def image(self, ctx, job, v, seed, dst=None):
        return draw(ctx, job, self.rows(v, stream(seed, self)), fx_values(0.0, 1.0, 0.0), clamp=True, dst=dst)


class Scratches2Legacy(LegacyNode):
    """Scattered scratches (ours): two sets of thin gaussian strands turned at random, the second four times as many,
    shorter and more varied in length: each set at random spots taken from a grid with twice as many cells
    (`spread`, fill 0.5: clumps and gaps), widths x Pattern size width / 100 and lengths x Pattern size height / 100;
    added on black and clamped. Sizes and gains fitted on the spectrum, histogram and correlations of the legacy
    node's outputs; Disorder slides each scratch along itself by up to Disorder / 100 of the image."""
    type = "scratches_scattered_legacy"
    label = "Scratches Scattered (Legacy)"
    params = {"pattern_size_width": Param(F, 100.0, "Pattern size width", 0.0, 100.0),
              "pattern_size_height": Param(F, 100.0, "Pattern size height", 0.0, 100.0),
              "disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}
    # Per set: count, width, shortest length, extra length (fraction), gain.
    SETS = ((2048, 0.00625, 0.234, 0.03, 0.35), (8192, 0.0078, 0.174, 0.4, 0.33))

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        w, h, out = 0.01 * v["pattern_size_width"], 0.01 * v["pattern_size_height"], []
        for k, width, length, extra, gain in self.SETS:
            out.append(strands(k, rng, w * width, (h * length, h * length * (1.0 + extra)), rng.random(k), gain,
                               0.01 * v["disorder"], at=spread(k, rng, 0.5)))
        return np.concatenate(out)

    def image(self, ctx, job, v, seed, dst=None):
        return draw(ctx, job, self.rows(v, stream(seed, self)), fx_values(0.0, 1.0, 0.0), clamp=True, dst=dst)


class Scratches4Legacy(LegacyNode):
    """Curved scratches (ours): COUNT thin crescents turned at random at random spots taken from a grid of about
    COUNT / FILL cells (`spread`: clumps and gaps), lightened (Max) on black in one pass, then softened by reading
    the image a fraction of a pixel off (bilinear); count, width, gain, clumping and softness fitted on the spectrum,
    histogram and correlations of the legacy node's outputs."""
    type = "scratches_curved_legacy"
    label = "Scratches Curved (Legacy)"
    params = {}
    COUNT, WIDTH, LENGTH, GAIN, FILL, SOFT = 3435, 0.0044, (0.094, 0.109), 1.3175, 0.44, 0.242

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        k = self.COUNT
        return strands(k, rng, self.WIDTH, self.LENGTH, rng.random(k), self.GAIN, pattern="Crescent",
                       at=spread(k, rng, self.FILL), blend=MAX)

    def image(self, ctx, job, v, seed, dst=None):
        img = draw(ctx, job, self.rows(v, stream(seed, self)), fx_values(0.0, 1.0, 0.0), clamp=True)
        return xf(ctx, job, img, IDENTITY, offset=(self.SOFT / job.size[0], self.SOFT / job.size[1]), dst=dst)



class ScratchesGeneratorLegacy(LegacyNode):
    """Scratches (ours): Scratches amount thin arcs at random spots and angles, bending either way by a sagitta of
    BEND of their length, half long (LONG x Scratches scale of the image, brightness LONG_GAIN) and half short
    (SHORT x Scratches scale, SHORT_GAIN), each SEGMENTS paraboloid strokes along its arc (OVERLAP times as long as
    their share, brightness tapering as a parabola along the scratch), ASPECT of its length wide but at least WIDTH
    pixels whatever the resolution (so their look changes with it), lightened (Max) on black; a scratch whose centre
    reads less than half Mask threshold on the Mask is left out (unconnected: every scratch). Lengths, bend,
    brightness, width and counts fitted on the legacy node's outputs (scratches measured one by one, means over the
    amount, scale and resolution)."""
    type = "scratches_generator_legacy"
    label = "Scratches generator (Legacy)"
    absolute_sizes = True
    inputs = {"mask": Input(PortType.GRAY, "Mask (Optional)", (1.0, 1.0, 1.0, 1.0))}
    params = {"scratches_amount": Param(ParamKind.INT, 2048, "Scratches amount", 0, 8192),
              "scratches_scale": Param(F, 0.2, "Scratches scale", 0.0, 1.0),
              "mask_threshold": Param(F, 0.5, "Mask threshold", 0.0, 1.0)}
    LONG, LONG_GAIN, SHORT, SHORT_GAIN = (0.607, 0.748), (0.45, 1.1), (0.18, 0.221), 0.29
    WIDTH, ASPECT, BEND, SEGMENTS, OVERLAP = 1.8, 0.015, 0.03, 3, 1.6

    def rows(self, v: dict, rng: np.random.Generator, width: float) -> np.ndarray:
        k, scale = v["scratches_amount"], v["scratches_scale"]
        long_ = np.arange(k) < k - k // 2
        length = scale * np.where(long_, rng.uniform(*self.LONG, k), rng.uniform(*self.SHORT, k))
        lum = np.where(long_, rng.uniform(*self.LONG_GAIN, k), self.SHORT_GAIN)
        centre, turn = rng.random((k, 2)), rng.random(k)
        sag = self.BEND * length * np.where(rng.random(k) < 0.5, -1.0, 1.0)
        along = np.stack([np.sin(2 * np.pi * turn), -np.cos(2 * np.pi * turn)], 1)  # the strokes' axis on screen
        across = np.stack([along[:, 1], -along[:, 0]], 1)
        n, out = self.SEGMENTS, []
        for i in range(n):
            x = (i - (n - 1) / 2) / n * length  # the segment's centre along the chord
            y = sag * (1.0 - 4.0 * x * x / length ** 2) - sag  # on the arc, the chord's middle at 0
            slope = np.arctan(-8.0 * sag * x / length ** 2) / (2 * np.pi)
            pos = centre + along * x[:, None] + across * y[:, None]
            size = np.stack([np.maximum(width, self.ASPECT * length), self.OVERLAP * length / n], 1)
            taper = 1.0 - 4.0 * x * x / length ** 2  # the scratch's brightness along it: a parabola
            out.append(stamps(pos, size, "Paraboloid", lum * taper, 0, MAX, rotation=turn - slope))
        return np.concatenate(out)

    def image(self, ctx, job, v, seed, dst=None):
        src = job.inputs.get("mask")
        cull = (ctx.ev.texture(*src), 0.5 * v["mask_threshold"]) if src else None
        rows = self.rows(v, stream(seed, self), self.WIDTH / job.size[0])
        return draw(ctx, job, rows, fx_values(0.0, 1.0, 0.0), clamp=True, dst=dst, cull=cull)

# --- Fluid ------------------------------------------------------------------------------------------------------------

class FluidLegacy(LegacyNode):
    """Falling drips (ours): the Fluid drips (noises_a/fluid.wgsl) at Scale 1, Pattern width x WIDTH[1] + WIDTH[0]
    wide and Pattern length x LENGTH long (in Fluid's Pattern Size), GAPS of the cells without a drip, wobbling with
    Warp intensity / 100 x WARP (in proportion to the width); Flow makes them fall Flow x FLOW cells each, a little
    more or less, and Disorder moves each by up to Disorder / 100 x JITTER / 2 cells per axis (a drip leaving its
    cell makes way for a new one). Fitted on the legacy node's outputs: histogram and correlation lengths over
    Pattern width and length, and the correlations from Warp intensity 0 to 50, Disorder 0 to 1, 5 and 10, Flow 0
    to 1, 5 and 10 (its shift too)."""
    type = "fluid_legacy"
    label = "Fluid (Legacy)"
    params = {**DISORDER,
              "warp_intensity": Param(F, 50.0, "Warp intensity", 0.0, 100.0),
              "flow": Param(F, 0.0, "Flow", 0.0, 100.0),
              "pattern_length": Param(F, 0.5, "Pattern length", 0.0, 1.0),
              "pattern_width": Param(F, 0.5, "Pattern width", 0.0, 1.0)}
    WIDTH, LENGTH, GAPS, WARP, FLOW, JITTER = (0.15, 0.9), 2.0, 0.25, 4.0, 1.22, 3.0

    def image(self, ctx, job, v, seed, dst=None):
        from trafilatura3d.nodes import NODES
        fluid = NODES["fluid"]
        width = self.WIDTH[0] + self.WIDTH[1] * v["pattern_width"]
        values = {**{k: p.default for k, p in fluid.params.items()}, "scale": 1, "disorder": 0.0,
                  "warp_intensity": 0.01 * v["warp_intensity"] * self.WARP * width / (self.WIDTH[0] + 0.5 * self.WIDTH[1]),
                  "pattern_size": [width, self.LENGTH * v["pattern_length"]], "gaps": self.GAPS,
                  "flow": self.FLOW * v["flow"], "jitter": 0.01 * v["disorder"] * self.JITTER}
        out = target(ctx, job, dst)
        ctx.dispatch(fluid, dataclasses.replace(job, seed=seed), values, dsts={"output": out})
        return out
