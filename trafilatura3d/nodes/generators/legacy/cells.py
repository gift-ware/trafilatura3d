"""The Legacy cellular noises and their family: Cells Domes / Mask / Walls / Flat, Crystal Facets / Shards, Creased,
Moisture noise, Plasma and Gaussian Spots Large / Small "(Legacy)", the versions old materials still use. All ours,
fitted on the outputs of the earlier legacy nodes: lattice noises (lattice.py: sites, domes, bells) and this module's
shaders (`cells_*.wgsl`), stamped shards (`shard_rows`), the Moisture noise shader. Cells Mask, Walls and Flat share
the sites of Cells Domes (Legacy) at their seed: their cells are its domes."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

from ._lib import F, I, LegacyNode, LegacyNoise, blur, fx_values, levels, stamps, target, warp
from .lattice import blobs, domes, sites
from ..noises_b.noises import MOISTURE_FIT, MoistureNoise1

MOISTURE = MoistureNoise1()


class _CellsPass(Node):
    """A pass of this module (its shader `<type>.wgsl` in this folder)."""
    registered = False
    outputs = {"output": Output(PortType.GRAY)}

    def wgsl(self) -> str:
        return (Path(__file__).parent / f"{self.type}.wgsl").read_text(encoding="utf-8")


def seeded(job, seed: float):
    """The job drawing at `seed` (the lattice shaders read the job's seed)."""
    return job if float(seed) == float(job.seed) else dataclasses.replace(job, seed=float(seed))


class Cells1Legacy(LegacyNode):
    """Domes cut into cells (ours, lattice.domes): on the (2 Distance + 1)^2 lattice of the tile one site per cell at
    SPREAD x u^EDGE steps from its centre (Disorder, in percent, turns them along a circle), a paraboloid RADIUS
    steps wide on each, overlaid with Max, black where none reaches; fitted on the spectrum, histogram and dark gaps
    of the legacy node's outputs."""
    type = "cells_domes_legacy"
    label = "Cells Domes (Legacy)"
    params = {"distance": Param(I, 14, "Distance", 0, 64), "disorder": Param(F, 50.0, "Disorder", 0.0, 100.0)}
    SPREAD, RADIUS, EDGE = 1.8, 1.6, 0.3

    def image(self, ctx, job, v, seed, dst=None):
        return domes(ctx, job, 2 * max(int(v["distance"]), 0) + 1, self.SPREAD, self.EDGE, self.RADIUS,
                     0.01 * v["disorder"], dst)


CELLS_1 = Cells1Legacy()


class _Mask(_CellsPass):
    type = "cells_legacy_mask"
    inputs = {"sites": Input(PortType.COLOR)}
    internal_uniforms = {"width": ParamKind.FLOAT, "width_px": ParamKind.FLOAT}


MASK = _Mask()


class Cells2Legacy(LegacyNode):
    """White cells with black walls (ours, cells_legacy_mask.wgsl): the regions nearest the sites of Cells Domes
    (Legacy) (its lattice and sites), the walls along their borders WIDTH cells plus WIDTH_PX pixels wide (thinner in
    cells at higher resolutions). Fitted on the coverage and spectrum of the legacy node's outputs at 256-2048."""
    type = "cells_mask_legacy"
    label = "Cells Mask (Legacy)"
    params = {"distance": Param(I, 14, "Distance", 10, 50), "disorder": Param(F, 50.0, "Disorder", 0.0, 100.0)}
    absolute_sizes = True
    WIDTH, WIDTH_PX = 0.17, 1.3

    def image(self, ctx, job, v, seed, dst=None):
        job = seeded(job, seed)
        c = CELLS_1
        s = sites(ctx, job, 2 * max(int(v["distance"]), 0) + 1, c.SPREAD, c.EDGE, 0.01 * v["disorder"])
        out = target(ctx, job, dst)
        ctx.dispatch(MASK, job, {"width": self.WIDTH, "width_px": self.WIDTH_PX}, srcs={"sites": (s, False)},
                     dsts={"output": out})
        return out, True


class _Walls(_CellsPass):
    type = "cells_legacy_walls"
    inputs = {"sites": Input(PortType.COLOR)}
    internal_uniforms = {"radius": ParamKind.FLOAT, "shrink": ParamKind.FLOAT, "width": ParamKind.FLOAT,
                         "depth": ParamKind.FLOAT}


WALLS = _Walls()


class Cells3Legacy(LegacyNode):
    """Thin walls on white (ours, cells_legacy_walls.wgsl): the domes of Cells Domes (Legacy) (its sites and radius)
    each pushed towards its top, so that a dome reads nearly white (SHRINK); where two domes meet, a line WIDTH
    pixels wide reads them at the two sites' midpoint: dark where the sites are far apart. Fitted on the walls'
    darkness, width and coverage of the legacy node's outputs (the walls stay about a pixel wide at every size)."""
    type = "cells_walls_legacy"
    label = "Cells Walls (Legacy)"
    params = {"distance": Param(I, 14, "Distance", 0, 128), "disorder": Param(F, 50.0, "Disorder", 0.0, 100.0)}
    SHRINK, WIDTH, DEPTH = 0.16, 0.7, 0.7

    def image(self, ctx, job, v, seed, dst=None):
        job = seeded(job, seed)
        c = CELLS_1
        s = sites(ctx, job, 2 * max(int(v["distance"]), 0) + 1, c.SPREAD, c.EDGE, 0.01 * v["disorder"])
        out = target(ctx, job, dst)
        ctx.dispatch(WALLS, job, {"radius": c.RADIUS, "shrink": self.SHRINK, "width": self.WIDTH,
                                  "depth": self.DEPTH},
                     srcs={"sites": (s, False)}, dsts={"output": out})
        return out, True


class _Flat(_CellsPass):
    type = "cells_legacy_flat"
    inputs = {"sites": Input(PortType.COLOR)}
    internal_uniforms = {"colors": ParamKind.INT, "grid": ParamKind.INT, "lo": ParamKind.FLOAT,
                         "hi": ParamKind.FLOAT, "power": ParamKind.FLOAT}


FLAT = _Flat()


class Cells4Legacy(LegacyNode):
    """Flat cells (ours, cells_legacy_flat.wgsl): the regions nearest the sites of Cells Domes (Legacy) (its lattice
    and sites), each one gray: one of Colors - 1 levels times an attenuation read at the site from a smooth random
    field over a lattice of n / SPAN cells (n the domes' lattice), cut between LO and HI and raised to POWER (dark and
    bright neighbourhoods, black patches). Fitted on the histogram and spectrum of the legacy node's outputs."""
    type = "cells_flat_legacy"
    label = "Cells Flat (Legacy)"
    params = {**Cells1Legacy.params, "colors": Param(I, 10, "Colors", 3, 20)}
    SPAN, LO, HI, POWER = 0.75, 0.24, 0.84, 1.0

    def image(self, ctx, job, v, seed, dst=None):
        job = seeded(job, seed)
        c = CELLS_1
        n = 2 * max(int(v["distance"]), 0) + 1
        s = sites(ctx, job, n, c.SPREAD, c.EDGE, 0.01 * v["disorder"])
        out = target(ctx, job, dst)
        ctx.dispatch(FLAT, job, {"colors": int(v["colors"]), "grid": max(round(n / self.SPAN), 1), "lo": self.LO,
                                 "hi": self.HI, "power": self.POWER},
                     srcs={"sites": (s, False)}, dsts={"output": out})
        return out, True


class _Crystal(_CellsPass):
    type = "cells_crystal"
    inputs = {"sites_a": Input(PortType.COLOR), "sites_b": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.GRAY), "product": Output(PortType.GRAY)}
    internal_uniforms = {"radius": ParamKind.FLOAT, "height": ParamKind.FLOAT, "gain": ParamKind.FLOAT}


CRYSTAL = _Crystal()


class Crystal1Legacy(LegacyNode):
    """Crystal facets (ours, cells_crystal.wgsl): two fields of domes on an N x N lattice from two independent sets
    of sites (lattice.sites: SPREAD x u^EDGE cells from the cells' centres, Disorder in percent turning them), each
    the Max of paraboloids RADIUS cells wide and HEIGHT tall; the output is GAIN x |A - B| (dark creases where the
    fields cross), warped by the slopes of A x B (Warp intensity x WARP). Fitted on the spectrum, histogram and
    creases of the legacy node's outputs."""
    type = "crystal_facets_legacy"
    label = "Crystal Facets (Legacy)"
    params = {"warp_intensity": Param(F, 0.0, "Warp intensity", 0.0, 100.0),
              "disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}
    N, SPREAD, EDGE, RADIUS, HEIGHT, GAIN, WARP = 16, 2.0, 0.4, 1.5, 0.91, 1.15, 0.006

    def image(self, ctx, job, v, seed, dst=None):
        job = seeded(job, seed)
        a = sites(ctx, job, self.N, self.SPREAD, self.EDGE, 0.01 * v["disorder"])
        b = sites(ctx, seeded(job, job.seed + 7919.0), self.N, self.SPREAD, self.EDGE, 0.01 * v["disorder"])
        w = self.WARP * v["warp_intensity"]
        out = target(ctx, job, None if w else dst)
        product = ctx.scratch(job, True, "32f")
        ctx.dispatch(CRYSTAL, job, {"radius": self.RADIUS, "height": self.HEIGHT, "gain": self.GAIN},
                     srcs={"sites_a": (a, False), "sites_b": (b, False)}, dsts={"output": out, "product": product})
        return warp(ctx, job, (out, True), (product, True), w, dst=dst) if w else (out, True)


def shard_rows(rng: np.random.Generator, count: int, top: float, octaves: float, aspect: tuple, tilt: float,
               cut: float, slope: float, gain: float) -> np.ndarray:
    """Stamp rows of `count` random shards (ours): uniform positions; heights `top` x 2^-(octaves x u), the number
    per octave growing 4x towards the small ones (sparse convolution with a scale-invariant density); widths
    `aspect` (a range) x the height; tilted up to `tilt` turns from vertical; Half Bells cut on one side by `cut`;
    signed amplitudes uniform in -1..1 x gain x (height / top)^slope (Add Sub)."""
    u = np.log1p(rng.random(count) * (4.0 ** octaves - 1.0)) / (octaves * math.log(4.0))
    h = top * 2.0 ** (-octaves * u)
    w = h * (aspect[0] + (aspect[1] - aspect[0]) * rng.random(count))
    lum = gain * (rng.random(count) * 2.0 - 1.0) * (h / top) ** slope
    return stamps(rng.random((count, 2)), np.stack([w, h], 1), "Half Bell", lum, 0,
                  rotation=1.0 + tilt * (rng.random(count) * 2.0 - 1.0), variation=cut)


class Crystal2Legacy(LegacyNoise):
    """Shards (ours): `shard_rows` stamped with Add Sub over a gray (BACKGROUND), cut below by Levels in low LOW, then
    warped by itself (Warp intensity x WARP). Fitted on the spectrum, histogram and look of the legacy node's
    outputs."""
    type = "crystal_shards_legacy"
    salt = 1639
    label = "Crystal Shards (Legacy)"
    params = {"warp_intensity": Param(F, 0.0, "Warp intensity", 0.0, 100.0)}
    BACKGROUND, LOW, WARP = 0.473, 0.17, 0.001
    SHARDS = {"count": 500, "top": 1.0, "octaves": 2.5, "aspect": (0.12, 0.5), "tilt": 0.07, "cut": 0.84,
              "slope": 1.5, "gain": 0.45}

    @property
    def FX(self):
        return fx_values(self.BACKGROUND, 1.0, 0.0)

    def rows(self, v, rng):
        return shard_rows(rng, **self.SHARDS)

    def finish(self, ctx, job, v, seed, img, dst):
        w = self.WARP * v["warp_intensity"]
        a = levels(ctx, job, img, low=self.LOW, dst=None if w else dst)
        return warp(ctx, job, a, a, w, dst=dst) if w else a


class _Creased(_CellsPass):
    type = "cells_creased"
    inputs = {"strokes": Input(PortType.GRAY)}
    internal_uniforms = {"shift": ParamKind.FLOAT, "fold": ParamKind.FLOAT}


CREASED = _Creased()


class CreasedLegacy(LegacyNoise):
    """Creases (ours): the shards of `shard_rows` (SHARDS) stamped with Add Sub over a gray (BACKGROUND), clamped,
    blurred (BLUR px at 256), then folded sideways by themselves (cells_creased.wgsl: the whole image moved by Warp
    intensity / 256, each pixel by FOLD x Warp intensity / 256 x the shards). Disorder moves each shard left by up to
    DISORDER x Disorder. Fitted on the spectrum, histogram and look of the legacy node's outputs."""
    type = "creased_legacy"
    label = "Creased (Legacy)"
    params = {"warp_intensity": Param(F, 64.0, "Warp intensity", 0.0, 128.0),
              "disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}
    BACKGROUND, BLUR, FOLD, DISORDER = 0.503, 3.0, 0.3, 0.001
    SHARDS = {**Crystal2Legacy.SHARDS, "gain": 0.4}

    @property
    def FX(self):
        return fx_values(self.BACKGROUND, 1.0, 0.0)

    def rows(self, v, rng):
        rows = shard_rows(rng, **self.SHARDS)
        rows[:, 0] -= self.DISORDER * v["disorder"] * rng.random(len(rows))
        return rows

    def finish(self, ctx, job, v, seed, img, dst):
        soft = blur(ctx, job, levels(ctx, job, img), self.BLUR)
        w = v["warp_intensity"] / 256.0
        out = target(ctx, job, dst)
        ctx.dispatch(CREASED, job, {"shift": w, "fold": self.FOLD * w}, srcs={"strokes": soft}, dsts={"output": out})
        return out, True


class MoistureNoiseLegacy(LegacyNode):
    """Soft blotches: the Moisture noise shader (noises_b/moisture_noise.wgsl, ours) with its own fit (FIT: 7 octaves
    from 8 cells across, no clustering, the contrast rising with the pattern's area), fitted on the spectrum, spread
    and mean of the legacy node's outputs. Pattern size width / height % stretch the blotches per axis; Disorder
    moves them along a loop."""
    type = "moisture_noise_legacy"
    label = "Moisture noise (Legacy)"
    params = {"pattern_size_width": Param(F, 100.0, "Pattern size width", 0.0, 1000.0),
              "pattern_size_height": Param(F, 100.0, "Pattern size height", 0.0, 1000.0),
              "disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}
    FIT = {**MOISTURE_FIT, "mo_rough": 0.74, "mo_octaves": 7, "mo_gain": 0.2, "mo_gain_up": 0.0, "mo_gain_up2": 0.0,
           "mo_area": 0.59, "mo_cluster_base": 1.0, "mo_cluster": 0.0, "mo_lift": 0.002}
    SCALE = 2  # the first octave 8 cells across

    def image(self, ctx, job, v, seed, dst=None):
        values = {**{k: prm.default for k, prm in MOISTURE.params.items()}, **self.FIT,
                  # the shader reads Pattern size across the image's y: this node's width is the x size
                  "pattern_size": [0.01 * v["pattern_size_height"], 0.01 * v["pattern_size_width"]],
                  "disorder": 0.01 * v["disorder"], "scale": self.SCALE, "mo_seed": float(int(seed) % 9973) * 0.731}
        out = target(ctx, job, dst)
        ctx.dispatch(MOISTURE, job, values, dsts={"output": out})
        return out, True


class _Plasma(_CellsPass):
    type = "cells_plasma"
    inputs = {"sites_a": Input(PortType.COLOR), "sites_b": Input(PortType.COLOR)}
    internal_uniforms = {"sigma": ParamKind.FLOAT, "contrast": ParamKind.FLOAT, "gain": ParamKind.FLOAT,
                         "gamma": ParamKind.FLOAT}


PLASMA = _Plasma()


class PlasmaLegacy(LegacyNode):
    """Plasma (ours, cells_plasma.wgsl): two smooth fields on the (2 Distance + 1)^2 lattice from two independent sets
    of sites (lattice.sites, SPREAD x u^EDGE cells from the cells' centres), each a sum of gaussian bells of sigma
    SIGMA cells, centred and scaled by CONTRAST around gray (clamped); the output is (GAIN x |A - B|)^GAMMA: bright
    where the two differ, dark veins where they cross. Fitted on the spectrum and histogram of the legacy node's
    outputs."""
    type = "plasma_legacy"
    label = "Plasma (Legacy)"
    params = {"distance": Param(I, 4, "Distance", 0, 64)}
    SPREAD, EDGE, SIGMA, CONTRAST, GAIN, GAMMA = 1.1, 0.3, 0.8, 0.28, 1.0, 1.0

    def image(self, ctx, job, v, seed, dst=None):
        job = seeded(job, seed)
        n = 2 * max(int(v["distance"]), 0) + 1
        a = sites(ctx, job, n, self.SPREAD, self.EDGE, 0.0)
        b = sites(ctx, seeded(job, job.seed + 7919.0), n, self.SPREAD, self.EDGE, 0.0)
        out = target(ctx, job, dst)
        ctx.dispatch(PLASMA, job, {"sigma": self.SIGMA, "contrast": self.CONTRAST, "gain": self.GAIN,
                                   "gamma": self.GAMMA},
                     srcs={"sites_a": (a, False), "sites_b": (b, False)}, dsts={"output": out})
        return out, True


class GaussianSpots1Legacy(LegacyNode):
    """Soft spots in clusters (ours, lattice.blobs): on a CELLS x CELLS lattice one gaussian bell of sigma SIGMA cells
    and random height per cell, its site SPREAD x u^EDGE cells from the centre plus an offset of up to GROUP cells
    shared by each 2 x 2 cells (chains and clumps), the heights modulated by a smooth field over 4 x 4 cells (PATCH:
    dense and empty regions), summed (GAIN) on black; Disorder (in percent) turns the sites.
    Fitted on the spectrum and histogram of the legacy node's outputs."""
    type = "gaussian_spots_large_legacy"
    label = "Gaussian Spots Large (Legacy)"
    params = {"disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}
    CELLS, SPREAD, EDGE, GROUP, PATCH, SIGMA, GAIN = 32, 0.3, 0.2, 1.2, 1.0, 0.33, 0.77

    def image(self, ctx, job, v, seed, dst=None):
        img = blobs(ctx, job, self.CELLS, self.SPREAD, self.EDGE, self.SIGMA, self.GAIN, 0.0, 0.01 * v["disorder"],
                    group=self.GROUP, random_height=True, patch=self.PATCH)
        return levels(ctx, job, img, dst=dst)


class _Spots(_CellsPass):
    type = "cells_spots"
    internal_uniforms = {"cells": ParamKind.INT, "sigma_lo": ParamKind.FLOAT, "sigma_hi": ParamKind.FLOAT,
                         "gain": ParamKind.FLOAT, "disorder": ParamKind.FLOAT}


SPOTS = _Spots()


class GaussianSpots2Legacy(LegacyNode):
    """Small soft spots (ours, cells_spots.wgsl): two lattices of CELLS and 2 x CELLS cells, one gaussian bell per
    cell at a random point inside it, its sigma random per axis in SIGMA cells (round and elongated spots), summed
    (GAIN) on black and clamped; Disorder (in percent) turns the bells around their cells. Fitted on the spectrum
    and histogram of the legacy node's outputs."""
    type = "gaussian_spots_small_legacy"
    label = "Gaussian Spots Small (Legacy)"
    params = {"disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}
    CELLS, SIGMA, GAIN = 64, (0.12, 0.22), 0.255

    def image(self, ctx, job, v, seed, dst=None):
        job = seeded(job, seed)
        out = target(ctx, job, dst)
        ctx.dispatch(SPOTS, job, {"cells": self.CELLS, "sigma_lo": self.SIGMA[0], "sigma_hi": self.SIGMA[1],
                                  "gain": self.GAIN, "disorder": 0.01 * v["disorder"]}, dsts={"output": out})
        return out, True
