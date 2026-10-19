"""Cells 1-4 and Crystal 1: cellular noises on one site per cell of a periodic Scale grid (Disorder
moves each site on a short closed path, so animating it loops). Cells 1, 3 and 4 share their sites
(cells_sites.wgsl). Crystal 2: random shards (Half Bells) over a continuous range of sizes (stamped.random_shards)."""

import numpy as np

from trafilatura3d.nodes.atomic.levels import Levels
from trafilatura3d.nodes.atomic.stamp_map import draw_patterns
from trafilatura3d.nodes.base import Input, Node, Param, PortType

from ..patterns.common import SHAPES_WGSL
from .common import (ANISO, CATEGORY, DISORDER, GRAY_OUT, HERE, NSE, NSE_CAPS, TILE_OFFSET, B, E, F, F2, I,
                     NoiseNode, scale)
from .stamped import FX_VALUES, random_shards

LEVELS = Levels()
SITES_WGSL = (HERE / "cells_sites.wgsl").read_text(encoding="utf-8")

# Cells 1's Pattern list, as its drop-down (cells_domes.wgsl maps them to shapes.wgsl).
PATTERNS = ("Paraboloid", "Bell", "Gaussian", "Thorn", "Half bell", "Cone", "Hemisphere")


class Cells1(NoiseNode):
    """On every cell of the Scale grid a pattern (the built-in shapes) 3 x Pattern size x Pattern scale cells
    wide, at its site (cells_sites.wgsl: anywhere in a disc around the cell's centre, Disorder turning it),
    overlaid with Max: walls where they meet, black gaps where none reaches."""
    type = "cells_domes"
    label = "Cells Domes"
    catalog = "cells_1"
    params = {
        **scale(32), **DISORDER, **ANISO,
        "pattern": Param(E, "Paraboloid", "Pattern", choices=PATTERNS),
        "pattern_size": Param(F2, [1.0, 1.0], "Pattern size", 0.0, 1.0),
        "pattern_scale": Param(F, 1.0, "Pattern scale", 0.0, 1.0),
        "luminance_random": Param(F, 0.0, "Luminance random", 0.0, 1.0),
        "angle": Param(F, 0.0, "Angle", 0.0, 1.0),
        "angle_random": Param(F, 0.5, "Angle random", 0.0, 1.0),
        **TILE_OFFSET, **NSE,
    }

    def wgsl(self) -> str:
        return SHAPES_WGSL + SITES_WGSL + super().wgsl()


class Cells2(NoiseNode):
    """Binary mask of cells (white) separated by walls 0.15 x Edge width cells wide."""
    type = "cells_mask"
    label = "Cells Mask"
    catalog = "cells_2"
    params = {**scale(32), "edge_width": Param(F, 2.0, "Edge width", 0.0, 4.0),
              "invert": Param(B, False, "Invert"), **DISORDER, **NSE}


class Cells3(NoiseNode):
    """Thin dark walls on white (ours, cells_walls.wgsl): where the domes of Cells 1 meet, from the two nearest
    sites (the bisector and the domes' slope jump there), darker and wider with Hardness."""
    type = "cells_walls"
    label = "Cells Walls"
    catalog = "cells_3"
    params = {
        **scale(32), "hardness": Param(F, 0.5, "Hardness", 0.0, 1.0), "invert": Param(B, False, "Invert"),
        **DISORDER, **ANISO,
        "pattern_size": Param(F2, [1.0, 1.0], "Pattern size", 0.0, 1.0),
        "pattern_scale": Param(F, 1.0, "Pattern scale", 0.0, 1.0),
        "angle": Param(F, 0.0, "Angle", 0.0, 1.0),
        "angle_random": Param(F, 0.5, "Angle random", 0.0, 1.0),
        **TILE_OFFSET, **NSE,
    }

    def wgsl(self) -> str:
        return SHAPES_WGSL + SITES_WGSL + super().wgsl()  # cells_sites.wgsl needs the shapes


class Cells4(NoiseNode):
    """A flat value per cell around the sites of Cells 1 (the region nearest each site): random (node seed),
    pseudorandom (own seed) or the input sampled at the cell's site."""
    type = "cells_flat"
    label = "Cells Flat"
    catalog = "cells_4"
    inputs = {"input": Input(PortType.GRAY, "Input", (0.5, 0.5, 0.5, 1.0))}
    params = {
        **scale(32), **DISORDER,
        "color_source": Param(E, "Random", "Color source",
                              choices=("Random", "Pseudorandom (Manual seed)", "Image input")),
        "pseudorandom_seed": Param(I, 0, "Pseudorandom seed", 0, 10),
        **NSE,
    }

    def wgsl(self) -> str:
        return SHAPES_WGSL + SITES_WGSL + super().wgsl()


class Crystal1(NoiseNode):
    """Faceted crystals (ours, crystal_facets.wgsl): the difference of the distances to two jittered point sets
    (two Worley fields), dark where they are equally near, bright facets elsewhere."""
    type = "crystal_facets"
    label = "Crystal Facets"
    catalog = "crystal_1"
    params = {**scale(32, 1, 256), "disorder": Param(F, 0.0, "Disorder", 0.0, 1.0), **NSE_CAPS}


class Crystal2(Node):
    """Shards (ours): `stamped.random_shards`, Half Bells cut on one side at uniform random positions over a
    continuous range of sizes (SHARDS), added to a gray (BACKGROUND), clamped. Fitted on the spectrum, histogram and
    look of the node's previous version."""
    type = "crystal_shards"
    label = "Crystal Shards"
    catalog = "crystal_2"
    category = CATEGORY
    outputs = GRAY_OUT
    params = {**scale(1, 1, 16), "disorder": Param(F, 0.0, "Disorder", 0.0, 1.0), **NSE_CAPS}
    BACKGROUND = 0.412
    SHARDS = {"count": 160, "top": 1.0, "octaves": 1.5, "aspect": (0.12, 0.5), "tilt": 0.06, "cut": 0.84,
              "slope": 1.5, "gain": 0.48}

    def rows(self, v: dict, size: tuple[int, int], seed: int) -> np.ndarray:
        rng = np.random.default_rng([seed & 0xFFFFFFFF, 2])
        return random_shards(v, size, rng, **self.SHARDS)

    def encode(self, ctx, job):
        g = self.BACKGROUND
        layer = ctx.scratch(job, True, "32f")
        draw_patterns(ctx, job, self.rows(job.values, job.size, int(job.seed)),
                      {**FX_VALUES, "background": [g, g, g, 1.0]}, dst=layer)
        # The atomic Levels at its defaults: the layer clamped to 0-1.
        ctx.dispatch(LEVELS, job, {k: prm.default for k, prm in LEVELS.params.items()}, srcs={"input": (layer, True)})
