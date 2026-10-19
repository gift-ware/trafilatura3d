"""Grunge Stains and Grunge Blots (Legacy), for the materials that still use them. Ours, each fitted on the spectrum,
histogram, quantiles and look of the legacy node's outputs: smooth fields (soft blobs and blotches, a blotchy field
swirled by the blobs: `fields`) set where the grime gathers; detail comes from smeared clouds, thin contour lines of
smooth fields (`ridge`), specks, scattered elongated and crescent marks (`marks`) and Gaussian spots' outlines (rings).
Both end in a common Brush pattern (a frayed rectangular patch, `grunge_patch.wgsl`) and Balance / Contrast
(`finish`)."""

from __future__ import annotations

from pathlib import Path

from trafilatura3d.nodes.atomic.directional_warp import DirectionalWarp
from trafilatura3d.nodes.base import Input, Node, Output, Param, PortType
from trafilatura3d.nodes.filters.basic.histogram_scan import HistogramScan

from ._lib import B, F, LegacyNode, apply, blur, dims, fit, levels, mix, rel, sized, warp, white
from .cells import GaussianSpots1Legacy, MoistureNoiseLegacy
from .clouds import Clouds1Legacy, Dirt1Legacy, Dirt3Legacy
from .noises import PerlinNoiseZoom
from .organic import DirectionnalNoiseLegacy
from .patterns import TileGeneratorLegacy

DWARP, SCAN = DirectionalWarp(), HistogramScan()
TILES, ZOOM, MOISTURE, CLOUDS_1 = TileGeneratorLegacy(), PerlinNoiseZoom(), MoistureNoiseLegacy(), Clouds1Legacy()
DIRT_1, DIRT_3, GAUSS_1, DIRECTIONAL = Dirt1Legacy(), Dirt3Legacy(), GaussianSpots1Legacy(), DirectionnalNoiseLegacy()
PARAMS = {"balance": Param(F, 0.5, "Balance", 0.0, 1.0), "contrast": Param(F, 0.5, "Contrast", 0.0, 1.0),
          "invert": Param(B, False, "Invert"), "brush_pattern": Param(F, 0.0, "Brush pattern", 0.0, 1.0)}


def tiles(ctx, job, seed: float, size, source=None, **values):
    """A Tile generator (Legacy) instance (Luminance variation 1, Position random 1)."""
    return TILES.instance(ctx, job, {"position_random": [1.0, 1.0], "luminance_variation_contrast": [1.0, 0.0],
                                     **values}, seed, size, source=source)


def dwarp(ctx, job, img, intensity_map, intensity: float, angle: float = 0.0):
    """Directional Warp at the input's size, the intensity map resampled to it."""
    j = sized(job, dims(img))
    return apply(ctx, j, DWARP, {"input": img, "intensity_input": fit(ctx, job, intensity_map, j.size)},
                 intensity=intensity, warp_angle=angle)


def ridge(ctx, job, img, t: float, w: float):
    """A thin bright line where `img` crosses `t` (1 at t, 0 beyond t +- w): its contour line."""
    up = levels(ctx, job, img, t - w, t + w)
    down = levels(ctx, job, img, t - w, t + w, out_low=1.0, out_high=0.0)
    return levels(ctx, job, mix(ctx, job, up, down, "Min"), 0.0, 0.5)


def fields(ctx, job, seed: float, swirl: float, distance: int):
    """The smooth fields both grunges start from: `zoom`, soft blobs over the tile at a quarter of the size (they bend
    everything else); `flow`, Moisture noise at half the size warped by them by `swirl` (blotches with swirled rings);
    `wide`, Moisture noise plus coarser blobs (a lattice of 2 `distance` + 1 cells) at a quarter: where grime gathers."""
    half, quarter = rel(job, -1), rel(job, -2)
    zoom = ZOOM.instance(ctx, job, {"distance": 5}, seed + 2, quarter)
    flow = warp(ctx, job, MOISTURE.instance(ctx, job, seed=seed + 1, size=half), zoom, swirl)
    wide = mix(ctx, job, MOISTURE.instance(ctx, job, seed=seed + 6, size=quarter),
               ZOOM.instance(ctx, job, {"distance": distance}, seed + 10, quarter), "Add sub", 0.5)
    return zoom, flow, wide


def marks(ctx, job, seed: float, zoom, size, pattern: str, mask: float):
    """Scattered marks: a Tile generator (Legacy) of thin `pattern`s on 7 x 7 cells, turned and placed at random, a
    random part of them (`mask`) left out, bent by the blobs."""
    t = tiles(ctx, job, seed, size, pattern_type=pattern, nb_x=7, nb_y=7, interstice=[0.8, 0.3, 0.0, 0.0],
              scale_variation=0.6, free_rotation_random=360.0, random_mask=mask, blending_mode="Max")
    return warp(ctx, job, t, zoom, 1.5)


class _Patch(Node):
    """The Brush pattern's patch (grunge_patch.wgsl)."""
    type = "grunge_patch"
    registered = False
    inputs = {"input": Input(PortType.GRAY)}
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"half_height": F, "soft": F, "fray": F}

    def wgsl(self) -> str:
        return (Path(__file__).parent / "grunge_patch.wgsl").read_text(encoding="utf-8")


PATCH = _Patch()


def finish(ctx, job, img, v: dict, inverted: bool, dst):
    """The common end: the image turned around when `inverted` (Stains is, unless Invert; Blots only with Invert);
    with Brush pattern, only a frayed rectangular patch of it is kept (shorter as Brush pattern grows, frayed more
    with Contrast); then Histogram Scan at Balance and Contrast."""
    flip = inverted != bool(v["invert"])
    a = levels(ctx, job, img, out_low=1.0 if flip else 0.0, out_high=0.0 if flip else 1.0)
    if v["brush_pattern"] > 0.0:
        a = apply(ctx, job, PATCH, {"input": a}, half_height=0.43 - 0.21 * v["brush_pattern"], soft=0.08,
                  fray=0.12 * v["brush_pattern"] * v["contrast"])
    return apply(ctx, job, SCAN, {"input": a}, dst, position=v["balance"], contrast=v["contrast"])


class GrungeMap009Legacy(LegacyNode):
    """Light grunge with dark stains (ours). Built as the grime's density D (0 clean, 1 black): creased clouds smeared
    by a directional warp along the blobs (hairy streaks) and levelled, times the gathering field (with a floor), plus
    a faint haze there; plus their contour lines, dark specks and a white noise grain where there is grime, minus fine
    white dust; then solid stains (the swirled blotches tilted by the gathering field, thresholded) and scattered dark
    strokes and crescents where grime is. D mapped to 0.75 .. 0.25 is what the common end (`finish`, inverted unless
    Invert) brings back to white .. black at the default Balance and Contrast. Fitted on the legacy node's outputs."""
    type = "grunge_stains_legacy"
    label = "Grunge Stains (Legacy)"
    params = PARAMS

    def image(self, ctx, job, v, seed, dst=None):
        g, half = job.size, rel(job, -1)
        zoom, flow, wide = fields(ctx, job, seed, 1.5, 4)
        density = levels(ctx, job, blur(ctx, job, wide, 4.0), 0.2, 0.5, out_low=1.0, out_high=0.0)
        hair = warp(ctx, job, dwarp(ctx, job, CLOUDS_1.instance(ctx, job, seed=seed + 3), zoom, 40.0), zoom, 1.0)
        tex = levels(ctx, job, hair, 0.25, 0.6, out_low=1.0, out_high=0.0)
        d = mix(ctx, job, tex, levels(ctx, job, density, out_low=0.25), "Multiply")
        d = mix(ctx, job, d, levels(ctx, job, wide, 0.2, 0.45, out_low=1.0, out_high=0.0), "Add", 0.15)
        d = mix(ctx, job, d, ridge(ctx, job, hair, 0.45, 0.01), "Add", mask=density)
        specks = levels(ctx, job, DIRT_1.instance(ctx, job, seed=seed + 5), 0.2, 0.6)
        d = mix(ctx, job, d, specks, "Add", 0.5, mask=density)
        d = mix(ctx, job, d, white(ctx, job, g), "Add sub", 0.6, mask=d)
        d = mix(ctx, job, d, DIRT_3.instance(ctx, job, seed=seed + 9), "Subtract", 0.8)
        stains = levels(ctx, job, mix(ctx, job, flow, wide, "Add sub", 0.35), 0.07, 0.1, out_low=1.0, out_high=0.0)
        d = mix(ctx, job, d, stains, "Max")
        strokes = levels(ctx, job, warp(ctx, job, marks(ctx, job, seed + 7, zoom, half, "Bell", 0.55), flow, 0.3),
                         0.15, 0.25)
        crescents = levels(ctx, job, marks(ctx, job, seed + 8, zoom, half, "Crescent", 0.5), 0.05, 0.15)
        m = mix(ctx, job, mix(ctx, job, strokes, crescents, "Max", 0.8), density, "Multiply")
        d = mix(ctx, job, d, levels(ctx, job, m, 0.2, 0.3), "Max")
        return finish(ctx, job, levels(ctx, job, d, out_low=0.25, out_high=0.75), v, True, dst)


class GrungeMap014Legacy(LegacyNode):
    """Gray grunge with dark blots (ours): the swirled blotches with a little of the gathering field, at full size;
    plus their contour lines; plus the outlines of Gaussian spots bent by the blobs (bright rings) minus a wider outline
    (their dark rims); minus contour lines of directional noise bent strongly by the blobs (streaks), only where a
    field of coarser blobs lets them; minus fine dust, plus specks in the light parts; times soft dark blots (the
    blotches tilted by the gathering field, thresholded down to 0.15); plus a light white noise grain; levelled to the
    legacy node's histogram, then the common end (`finish`, inverted with Invert). Fitted on the legacy node's
    outputs."""
    type = "grunge_blots_legacy"
    label = "Grunge Blots (Legacy)"
    params = PARAMS

    def image(self, ctx, job, v, seed, dst=None):
        g, half, quarter = job.size, rel(job, -1), rel(job, -2)
        zoom, flow, wide = fields(ctx, job, seed, 2.0, 3)
        out = mix(ctx, job, flow, wide, "Add sub", 0.2, size=g)
        out = mix(ctx, job, out, ridge(ctx, job, fit(ctx, job, flow, g), 0.5, 0.012), "Add", 0.5)
        spots = warp(ctx, job, GAUSS_1.instance(ctx, job, seed=seed + 4), zoom, 0.5)
        out = mix(ctx, job, out, ridge(ctx, job, spots, 0.3, 0.025), "Add", 0.5)
        out = mix(ctx, job, out, ridge(ctx, job, spots, 0.45, 0.025), "Subtract", 0.4)
        streaks = warp(ctx, job, fit(ctx, job, DIRECTIONAL.instance(ctx, job, seed=seed + 3, size=half), g), zoom, 4.0)
        streaks = ridge(ctx, job, blur(ctx, job, streaks, 0.3), 0.3, 0.03)
        where = levels(ctx, job, ZOOM.instance(ctx, job, {"distance": 3}, seed + 11, quarter), 0.35, 0.6)
        out = mix(ctx, job, out, streaks, "Subtract", mask=where)
        out = mix(ctx, job, out, DIRT_3.instance(ctx, job, seed=seed + 9), "Subtract", 0.15)
        out = mix(ctx, job, out, DIRT_1.instance(ctx, job, seed=seed + 5), "Add", 0.7,
                  mask=levels(ctx, job, wide, 0.3, 0.6))
        blots = levels(ctx, job, mix(ctx, job, flow, wide, "Add sub", 0.5), 0.02, 0.25, out_low=0.15)
        out = mix(ctx, job, out, blots, "Multiply")
        out = mix(ctx, job, out, white(ctx, job, g), "Add sub", 0.1)
        return finish(ctx, job, levels(ctx, job, out, 0.009, 0.926, out_low=0.25, out_high=0.75), v, False, dst)
