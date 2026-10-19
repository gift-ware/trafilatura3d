"""Legacy pattern generators that differ from the current ones (Tile Generator, Weave Twill, Waveform; the
documentation doesn't list them). Each is an FX-Map whose patterns are generated here (numpy rows) and drawn by the
FX-Map engine (atomic/stamp_map.py `draw_patterns`), then adjusted with the atomic nodes. The other Legacy patterns
are the current nodes without Non Square Expansion: the importer maps them onto the current nodes. Random values come
from a numpy stream seeded by the job (`seeded_random` for the per-tile values)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from trafilatura3d.fxmap import PATTERNS
from trafilatura3d.nodes.atomic.stamp_map import MAX_INPUTS
from trafilatura3d.nodes.base import Input, Node, Output, Param, PortType

from ._lib import ADD_SUB, MAX, B, E, F, F2, F4, I, LegacyNode, draw, fx_values, stamps, stream, target

HERE = Path(__file__).parent
SHAPES_WGSL = (HERE.parent.parent / "shapes.wgsl").read_text(encoding="utf-8")  # the built-in pattern shapes (`shape()`)


POSITION_SPREAD = 0.577  # Position random: a uniform move of up to this x the value, in cells (spread fitted)
OFFSET_SPREAD = 0.55  # Offset random: a line's shift of up to this x the value, of the image (fitted)


def seeded_random(index, seed: int) -> np.ndarray:
    """A value in [0, 1) per integer `index`, the same for the same (index, seed) whatever the node's seed (a PCG
    hash, as the shaders' `pcg`)."""
    x = (np.asarray(index, np.int64) * 747796405 + seed * 2891336453 + 12345) & 0xFFFFFFFF
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737 & 0xFFFFFFFF
    return (((x >> 22) ^ x) & 0xFFFFFFFF) / 4294967296.0


def middle_widths(m: float, n: int) -> tuple[float, float]:
    """Middle size along an axis of `n` lines: the widths in uv of the even and the odd lines (ceil(n / 2) even lines,
    floor(n / 2) odd ones, packed)."""
    even, odd = math.ceil(n / 2), n // 2
    we = m * 2.0 / n if m <= 0.5 else (1.0 - (1.0 - m) * 2.0 * odd / n) / max(even, 1)
    return we, (1.0 - we * even) / max(odd, 1)


def line_starts(i: np.ndarray, we: float, wo: float) -> np.ndarray:
    """Where line `i` starts: the even and odd lines before it."""
    return np.ceil(i / 2) * we + np.floor(i / 2) * wo


SHAPE_CHOICES = ("Image input", "Square", "Disc", "Paraboloid", "Bell", "Gaussian", "Thorn", "Pyramid", "Brick",
                 "Gradation", "Waves", "Half bell", "Ridged bell", "Crescent", "Capsule", "Cone")  # FX index - 1


class TileGeneratorLegacy(LegacyNode):
    """Number X x Number Y tiles: one FX-Map pass drawing a pattern per tile, tile k in row k // Number X,
    column k % Number X. Columns and rows alternate the widths of Middle size
    (even lines 2 x Middle size / count of the image, odd lines the rest, packed); a tile is its line's width less
    Interstice (x, x random, y, y random: the random part gives back up to x of the gap per tile) times Scale, times
    1 - Scale variation + Scale variation x seeded_random(k, seed) (the same factor as Luminance by scale variation).
    Rotation in turns = Free rotation / 360 + Free rotation random / 360 x U[-1, 1) + Fixed rotation / 4 + half a
    turn with probability Fixed rotation random; an odd Fixed rotation swaps the tile's size (it keeps the cell's
    footprint). Quincunx flip instead adds a quarter turn per step of the 2 x 2 block (column parity + 2 x row
    parity) and swaps the size where column + row is odd. Position: the cell's centre, line k (the row, or the column
    with Vertical offset) shifted by Offset x k cells plus Offset random x OFFSET_SPREAD x seeded_random(k, seed) of
    the image, plus Position random x POSITION_SPREAD x U[-1, 1) cells per axis, plus Global offset. Luminance:
    lerp(1, rand, variation), pushed away from 0.5 by 1 / (1 - contrast / 2) (clamped to -1..1), times k / tiles with Luminance by number, times the masks
    (Random: rand >= Random mask; Checker: even column + row; Horizontal: even rows; Vertical: even columns;
    Invert flips them all); Add sub draws it minus Background color (signed: masked tiles subtract it), Add and Max
    clamped to 0-1, over Background color (or the Background input) times Global opacity. Reverse rendering order
    places tile k where tile tiles - 1 - k would be (sizes and luminance keep k). The legacy list has
    Add, Max and Add sub; its Max is the FX-Map's Max (the gray FX-Map has no alpha to blend)."""
    type = "tile_generator_legacy"
    label = "Tile generator (Legacy)"
    inputs = {"source": Input(PortType.GRAY, "Pattern image input"), "background": Input(PortType.GRAY, "Background")}
    params = {
        "nb_x": Param(I, 10, "Number X", 1, 64),
        "nb_y": Param(I, 10, "Number Y", 1, 64),
        "middle_size": Param(F2, [0.5, 0.5], "Middle size", 0.0, 1.0),
        "scale": Param(F, 1.0, "Scale", 0.0, 2.0),
        "scale_variation": Param(F, 0.0, "Scale variation", 0.0, 1.0),
        "scale_variation_seed": Param(I, 0, "Scale variation seed", 0, 1000),
        "luminance_by_scale_variation": Param(B, False, "Luminance by scale variation"),
        "interstice": Param(F4, [0.0, 0.0, 0.0, 0.0], "Interstice", 0.0, 1.0),
        "offset": Param(F, 0.0, "Offset", 0.0, 3.0),
        "offset_random": Param(F, 0.0, "Offset random", 0.0, 1.0),
        "offset_random_seed": Param(I, 0, "Offset random seed", 0, 1000),
        "vertical_offset": Param(B, False, "Vertical offset"),
        "global_offset": Param(F2, [0.0, 0.0], "Global offset", 0.0, 1.0),
        "luminance_by_number": Param(B, False, "Luminance by number"),
        "luminance_variation_contrast": Param(F2, [0.0, 0.0], "Luminance variation/contrast", 0.0, 1.0),
        "pattern_type": Param(E, "Brick", "Pattern type", choices=SHAPE_CHOICES),
        "pattern_specific": Param(F2, [0.2, 0.0], "Pattern specific", 0.0, 1.0),
        "image_input_filtering": Param(E, "Bilinear + mipmaps", "Image input filtering (Engine >v4 only)",
                                       choices=("Bilinear + mipmaps", "Bilinear", "Nearest")),
        "free_rotation": Param(F, 0.0, "Free rotation", 0.0, 360.0),
        "free_rotation_random": Param(F, 0.0, "Free rotation random", 0.0, 360.0),
        "fixed_rotation": Param(I, 0, "Fixed rotation", 0, 3),
        "fixed_rotation_random": Param(F, 0.0, "Fixed rotation random", 0.0, 1.0),
        "quincunx_flip": Param(B, False, "Quincunx flip"),
        "position_random": Param(F2, [0.0, 0.0], "Position random", 0.0, 10.0),
        "checker_mask": Param(B, False, "Checker mask"),
        "horizontal_mask": Param(B, False, "Horizontal mask"),
        "vertical_mask": Param(B, False, "Vertical mask"),
        "random_mask": Param(F, 0.0, "Random mask", 0.0, 1.0),
        "invert_mask": Param(B, False, "Invert mask"),
        "blending_mode": Param(E, "Add", "Blending mode", choices=("Add", "Max", "Add sub")),
        "background_color": Param(F, 0.0, "Background color", 0.0, 1.0),
        "global_opacity": Param(F, 1.0, "Global opacity", 0.0, 1.0),
        "reverse_rendering_order": Param(B, False, "Reverse rendering order"),
    }

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        nx, ny = max(int(v["nb_x"]), 1), max(int(v["nb_y"]), 1)
        count = nx * ny
        k = np.arange(count)
        row, col = k // nx, k % nx  # tiles numbered row by row
        t = count - 1 - k if v["reverse_rendering_order"] else k  # where tile k is placed
        trow, tcol = t // nx, t % nx
        wex, wox = middle_widths(v["middle_size"][0], nx)
        wey, woy = middle_widths(v["middle_size"][1], ny)
        # Size: the line's width less the interstice, swapped by the rotation, times Scale and its variation.
        ix, ixr, iy, iyr = v["interstice"]
        w = np.where(col % 2 == 0, wex, wox) * (1 - ix * (1 - ixr * rng.random(count)))
        h = np.where(row % 2 == 0, wey, woy) * (1 - iy * (1 - iyr * rng.random(count)))
        swap = (col + row) % 2 == 1 if v["quincunx_flip"] else np.full(count, int(v["fixed_rotation"]) % 2 == 1)
        sv = v["scale_variation"]
        by_scale = 1 - sv + sv * seeded_random(k, int(v["scale_variation_seed"]))
        size = np.where(swap[:, None], np.stack([h, w], 1), np.stack([w, h], 1)) * (v["scale"] * by_scale)[:, None]
        # Rotation in turns.
        rot = (v["free_rotation"] + v["free_rotation_random"] * (2 * rng.random(count) - 1)) / 360.0 \
            + int(v["fixed_rotation"]) / 4.0 + np.where(rng.random(count) >= v["fixed_rotation_random"], 0.0, 0.5)
        if v["quincunx_flip"]:
            rot += 0.25 * ((col % 2) + 2 * (row % 2))  # a quarter turn more at each step of the 2 x 2 block
        # Position: the cell's centre in its line, the line's offset, the random move (each line shifted by up to
        # Offset random x OFFSET_SPREAD of the image, the same for every line; spread fitted on the output).
        off, orand, oseed = v["offset"], v["offset_random"], int(v["offset_random_seed"])
        shift_x = shift_y = 0.0
        if v["vertical_offset"]:
            shift_y = orand * OFFSET_SPREAD * seeded_random(tcol, oseed) + off / ny * tcol
        else:
            shift_x = orand * OFFSET_SPREAD * seeded_random(trow, oseed) + off / nx * trow
        x = line_starts(tcol, wex, wox) + 0.5 * np.where(tcol % 2 == 0, wex, wox) + shift_x
        y = line_starts(trow, wey, woy) + 0.5 * np.where(trow % 2 == 0, wey, woy) + shift_y
        prx, pry = v["position_random"]
        x = x + prx * POSITION_SPREAD * (2 * rng.random(count) - 1) / nx
        y = y + pry * POSITION_SPREAD * (2 * rng.random(count) - 1) / ny
        pos = np.stack([x, y], 1) + np.asarray(v["global_offset"])
        # Luminance: variation (lerp from 1 to a random value), contrast pushing it away from mid gray, by number,
        # masks, by scale.
        lvx, lvy = v["luminance_variation_contrast"]
        var = 1 + lvx * (rng.random(count) - 1)
        lum = np.clip(0.5 + (var - 0.5) / (1 - 0.5 * lvy), -1.0, 1.0)  # contrast: x2 at 1
        if v["luminance_by_number"]:
            lum = lum * k / count
        mask = (rng.random(count) >= v["random_mask"]).astype(np.float64)
        if v["checker_mask"]:
            mask *= (col + row) % 2 == 0
        if v["horizontal_mask"]:
            mask *= row % 2 == 0
        if v["vertical_mask"]:
            mask *= col % 2 == 0
        if v["invert_mask"]:
            mask = 1 - mask
        lum = mask * lum * (by_scale if v["luminance_by_scale_variation"] else 1.0)
        add_sub = v["blending_mode"] == "Add sub"
        lum = lum - v["background_color"] if add_sub else np.clip(lum, 0.0, 1.0)
        psx, psy = v["pattern_specific"]
        pattern = PATTERNS[SHAPE_CHOICES.index(v["pattern_type"]) + 1]
        out = stamps(pos, size, pattern, lum, 0, MAX if v["blending_mode"] == "Max" else ADD_SUB, rotation=rot,
                     variation=psx * (1 - psy + psy * rng.random(count)))
        out[:, 12] = v["image_input_filtering"] == "Nearest"
        return out

    def image(self, ctx, job, v, seed, dst=None, source=None):
        """`source`: the Pattern image input when drawn as a part of another node (default: the job's input)."""
        src = job.inputs.get("source")
        if source is not None:
            pattern = source[0]
        else:
            pattern = ctx.ev.texture(*src) if src else ctx.ev.constant((0.0, 0.0, 0.0, 0.0))
        images = [pattern] + [ctx.ev.constant((0.0, 0.0, 0.0, 0.0))] * (MAX_INPUTS - 1)
        # Clamped, as a 16 bit FX-Map.
        return draw(ctx, job, self.rows(v, stream(seed, self)), fx_values(v["background_color"], v["global_opacity"]),
                    levels=1, clamp=True, images=images, dst=dst)


@dataclasses.dataclass(frozen=True)
class Floats:
    """A family of thread floats of Weave Twill (Legacy), in cells: the threads' direction (radians), the float's half
    length and half width, its profile's exponents along and across, its bend (sideways at the ends), its gain, and
    its centre's place in the cell."""
    angle: float
    half_length: float
    half_width: float
    along: float
    across: float
    bend: float
    gain: float
    place: tuple[float, float]

    def uniforms(self, n: int) -> dict:
        reach = math.hypot(self.half_length, self.half_width + abs(self.bend))
        return {f"shape{n}": [math.cos(self.angle), math.sin(self.angle), self.half_length, self.half_width],
                f"profile{n}": [self.along, self.across, self.bend, self.gain],
                f"place{n}": [self.place[0] % 1.0, self.place[1] % 1.0, reach, 0.0]}


class _WeaveMoves(Node):
    """The floats' moves by Disorder, one texel per cell (patterns_weave_moves.wgsl)."""
    type = "patterns_weave_moves"
    registered = False
    outputs = {"output": Output(PortType.COLOR)}
    internal_uniforms = {"disorder": F}

    def wgsl(self) -> str:
        return (HERE / "patterns_weave_moves.wgsl").read_text(encoding="utf-8")


class _WeaveFloats(Node):
    """The weave of Weave Twill (Legacy), computed per pixel (patterns_weave.wgsl)."""
    type = "patterns_weave"
    registered = False
    inputs = {"moves": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"tiles": I, "disorder": F, "rotate": I, "lift": F, "tone0": F4, "tone1": F4, "tone2": F4,
                         **{f"{k}{n}": F4 for n in (1, 2) for k in ("shape", "profile", "place")}}

    def wgsl(self) -> str:
        return (HERE / "patterns_weave.wgsl").read_text(encoding="utf-8")


WEAVE_MOVES, WEAVE_FLOATS = _WeaveMoves(), _WeaveFloats()


class Weave2Legacy(LegacyNode):
    """Woven threads, ours (patterns_weave.wgsl): a twill of two families of thread floats on 16 x 16 cells per tile
    (Tiling tiles across), one float of each family per cell, each an arc along its thread, bent and rounded across
    (`Floats`); the floats add up, then a lift and a tone curve (TONE). Disorder moves every float by up to DISORDER_CELLS x
    Disorder cells at a random angle (the same in every tile); Rotate 45 degrees (on by default) samples the weave
    turned by 45 degrees and enlarged by sqrt 2. Fitted on the legacy node's outputs (its cell pixel by pixel, the
    statistics with Disorder)."""
    type = "weave_twill_legacy"
    label = "Weave Twill (Legacy)"
    inputs = {}
    params = {
        "tiling": Param(I, 1, "Tiling", 1, 16),
        "disorder": Param(F, 0.0, "Disorder", 0.0, 100.0),
        "rotate_45_degrees": Param(B, True, "Rotate 45 degrees"),
    }
    WARP = Floats(1.092, 1.558, 0.3002, 1.819, 1.598, 0.1126, 1.333, (0.7191, 0.3849))
    WEFT = Floats(0.01606, 1.255, 0.5771, 2.457, 2.216, 0.04468, 0.2522, (0.8805, 0.0006531))
    LIFT = -0.3616
    # The tone curve at sums -0.2, 0, 0.2 ... 2.0 (fitted on the legacy node's value distribution: a gradual approach
    # to white).
    TONE = (0.0, 0.0, 0.0281, 0.1722, 0.3107, 0.5947, 0.8948, 1.0, 1.0, 1.0, 1.0, 1.0)
    DISORDER_CELLS = 0.016

    def image(self, ctx, job, v, seed, dst=None):
        out = target(ctx, job, dst)
        seeded = dataclasses.replace(job, seed=float(seed))
        disorder = self.DISORDER_CELLS * v["disorder"]
        moves = ctx.scratch(job, False, "32f", size=(16, 16))
        ctx.dispatch(WEAVE_MOVES, seeded, {"disorder": disorder}, dsts={"output": moves}, size=(16, 16))
        values = {"tiles": max(int(v["tiling"]), 1), "disorder": disorder,
                  "rotate": int(bool(v["rotate_45_degrees"])), "lift": self.LIFT, **{f"tone{i}": list(self.TONE[4 * i:4 * i + 4]) for i in range(3)},
                  **self.WARP.uniforms(1), **self.WEFT.uniforms(2)}
        ctx.dispatch(WEAVE_FLOATS, seeded, values, srcs={"moves": (moves, False)}, dsts={"output": out})
        return out, True


class _WaveformBars(Node):
    """The bars of Waveform (Legacy), computed per pixel (patterns_waveform.wgsl after the built-in shapes)."""
    type = "patterns_waveform"
    registered = False
    outputs = {"output": Output(PortType.GRAY)}
    internal_uniforms = {"samples": I, "size_min": F, "size_max": F, "wave_number": I, "noise": F, "kind": I}

    def wgsl(self) -> str:
        return SHAPES_WGSL + (HERE / "patterns_waveform.wgsl").read_text(encoding="utf-8")


WAVEFORM_BARS = _WaveformBars()


class Waveform1Legacy(LegacyNode):
    """Samples bars side by side, ours (patterns_waveform.wgsl): bar k fills its column, centred on the middle row,
    SizeMax x lerp(F, 1, SizeMin) tall with F = W x (1 - Noise + Noise x a random value per bar) and W a raised cosine
    of WaveNumber periods, 0.5 - 0.5 cos(2 pi WaveNumber (k + 1) / Samples) (1 with WaveNumber 0); along its height
    the bar is the Pattern (the built-in shapes by FX-Map index, 2 Square ... 15 Capsule; default 6 Gaussian) doubled
    and clamped, on black; an output narrower than Samples averages the bars under each pixel. Fitted on the legacy
    node's outputs (bars, profiles, statistics)."""
    type = "waveform_legacy"
    label = "Waveform (Legacy)"
    params = {
        "samples": Param(I, 256, "Samples", 0, 1024),
        "size_min": Param(F, 0.0, "SizeMin", 0.0, 1.0),
        "size_max": Param(F, 1.0, "SizeMax", 0.0, 1.0),
        "wave_number": Param(I, 2, "WaveNumber", 0, 64),
        "noise": Param(F, 0.5, "Noise", 0.0, 1.0),
        "pattern": Param(I, 6, "Pattern", 2, 15),
    }

    def image(self, ctx, job, v, seed, dst=None):
        out = target(ctx, job, dst)
        values = {k: v[k] for k in ("samples", "size_min", "size_max", "wave_number", "noise")}
        ctx.dispatch(WAVEFORM_BARS, dataclasses.replace(job, seed=float(seed)),
                     {**values, "kind": int(np.clip(v["pattern"], 2, 15)) - 2}, dsts={"output": out})
        return out, True


class SplatterCircularLegacy(LegacyNode):
    """Number patterns on a circle about the centre, pattern i at the angle i / Number of a turn counterclockwise
    from the right, less Position random x a random fraction of a radian; at Radius x (1 - Radius random x a
    random fraction) from the centre. Each is Pattern size x Pattern scale x (1 - Pattern scale random x f) wide (f
    random), each axis also x (1 - Pattern size random x its own random fraction); turned Rotation + Rotation random x
    a random fraction of a turn clockwise, and with Center orientation back by its angle (its x axis along the
    radius); Pattern pivot Min / Max moves it half its width along its x axis forwards / backwards. Luminance:
    (1 - Luminance random x a random fraction) x (1 - Luminance by scale x Pattern scale random x f); a pattern is
    left out with probability Random mask. Drawn in one FX-Map pass (Add sub or Max) over the Background input,
    clamped; Pattern image input is the Image input pattern. Every rule measured on the legacy node's outputs
    (patterns found one by one: centres, sizes, orientations, luminance, over the parameters and many seeds)."""
    type = "splatter_circular_legacy"
    label = "Splatter circular (Legacy)"
    inputs = {"source": Input(PortType.GRAY, "Pattern image input"), "background": Input(PortType.GRAY, "Background")}
    params = {
        "number": Param(I, 12, "Number", 1, 64),
        "radius": Param(F, 0.25, "Radius", 0.0, 1.0),
        "radius_random": Param(F, 0.0, "Radius random", 0.0, 1.0),
        "pattern": Param(E, "Square", "Pattern", choices=SHAPE_CHOICES),
        "image_input_filtering": Param(E, "Bilinear + mipmaps", "Image input filtering (Engine v5 only)",
                                       choices=("Bilinear + mipmaps", "Bilinear", "Nearest")),
        "pattern_specific": Param(F, 0.0, "Pattern specific", 0.0, 1.0),
        "pattern_size": Param(F2, [0.1, 0.1], "Pattern size", 0.0, 1.0),
        "pattern_size_random": Param(F, 0.0, "Pattern size random", 0.0, 1.0),
        "pattern_scale": Param(F, 1.0, "Pattern scale", 0.0, 2.0),
        "pattern_scale_random": Param(F, 0.0, "Pattern scale random", 0.0, 1.0),
        "position_random": Param(F, 0.0, "Position random", 0.0, 1.0),
        "random_mask": Param(F, 0.0, "Random mask", 0.0, 1.0),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0),
        "rotation_random": Param(F, 0.0, "Rotation random", 0.0, 1.0),
        "center_orientation": Param(B, False, "Center orientation"),
        "pattern_pivot": Param(E, "Center", "Pattern pivot", choices=("Center", "Min", "Max")),
        "blending_mode": Param(E, "Add sub", "Blending mode", choices=("Add sub", "Max")),
        "luminance_random": Param(F, 0.0, "Luminance random", 0.0, 1.0),
        "luminance_by_scale": Param(F, 1.0, "Luminance by scale", 0.0, 1.0),
    }

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        n = int(v["number"])
        u = rng.random((n, 7))
        theta = 2 * np.pi * np.arange(n) / n - v["position_random"] * u[:, 0]
        r = v["radius"] * (1.0 - v["radius_random"] * u[:, 1])
        pos = np.stack([0.5 + r * np.cos(theta), 0.5 - r * np.sin(theta)], 1)
        f = 1.0 - v["pattern_scale_random"] * u[:, 2]
        size = np.asarray(v["pattern_size"]) * v["pattern_scale"] * f[:, None] * \
            (1.0 - v["pattern_size_random"] * u[:, 3:5])
        rot = v["rotation"] + v["rotation_random"] * u[:, 5] - (theta / (2 * np.pi) if v["center_orientation"] else 0.0)
        pivot = {"Center": 0.0, "Min": 0.5, "Max": -0.5}[v["pattern_pivot"]] * size[:, 0]
        pos = pos + pivot[:, None] * np.stack([np.cos(2 * np.pi * rot), np.sin(2 * np.pi * rot)], 1)
        lum = (1.0 - v["luminance_random"] * rng.random(n)) * (1.0 - v["luminance_by_scale"] * (1.0 - f))
        pattern = PATTERNS[SHAPE_CHOICES.index(v["pattern"]) + 1]
        rows = stamps(pos, size, pattern, lum, 0, MAX if v["blending_mode"] == "Max" else ADD_SUB, rotation=rot,
                      variation=v["pattern_specific"])
        rows[:, 12] = float(v["image_input_filtering"] == "Nearest")
        return rows[u[:, 6] >= v["random_mask"]]

    def image(self, ctx, job, v, seed, dst=None):
        src = job.inputs.get("source")
        images = [ctx.ev.texture(*src) if src else ctx.ev.constant((0.0, 0.0, 0.0, 0.0))] + \
            [ctx.ev.constant((0.0, 0.0, 0.0, 0.0))] * (MAX_INPUTS - 1)
        return draw(ctx, job, self.rows(v, stream(seed, self)), fx_values(0.0, 1.0, 0.0), levels=1, clamp=True,
                    images=images, dst=dst)
