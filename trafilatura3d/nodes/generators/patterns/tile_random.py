import struct
from math import ceil

import numpy as np

from .common import (B, C, E, F, F4, FILTERING, I, Input, Output, Param, PatternNode, PortType, SYMMETRY,
                     color_variant, nse, pattern_choices)

ROTATIONS = ("0", "90", "180", "270", "random horizontal", "random vertical")
SPLIT = ("none", "auto", "auto horizontal", "auto vertical", "random h+v")
INTERSTICE = ("Relative to smallest brick", "Relative to largest brick")
PARAMETRISATION = ("none", "area", "size x", "size y")
AFFECT = ("RGB+Alpha", "RGB only", "Alpha only")
BLENDING = ("Add/Sub", "Max", "Alpha Blend")
DATA_WIDTH = 1024  # texels per row of the uploaded layout


class TileRandom(PatternNode):
    """Tile Random (laid out on the CPU, drawn by a compute pass). Y Amount rows of X Amount
    bricks: the boundaries are jittered by up to half a brick (Random Y, Random X), the first and last of each line
    stay on the image border; Offset shifts row j by j x Offset bricks (a staircase), Offset Random each row by up to
    its value in uv. Each brick is cut into N pieces across its longer side, N = ceil(aspect + Multiplier) when its
    short/long ratio is below 2 x Threshold (else 1 + Multiplier); each piece is cut again by the same rule with
    max(2 x Threshold, 1) - 1 and no Multiplier (Split Mode: auto, only across the width / the height, a random axis;
    the cuts are jittered by Random X/Y). A first pass measures the smallest and largest leaves: the interstice is
    Amount^2 x the smaller side of the smallest or largest leaf. Each leaf draws its pattern (a shape or the Pattern
    Input) scaled (lerp(Scale, random, Scale Random)), rotated (Rotation +- Rotation Random turns; Rotation Constraint
    shrinks a rotated tile), moved (Random Constraints: by +-half the free space times Random, else up to Random / 2
    in uv), with 90-degree pattern steps (a random step with probability Rotation Random) and random mirroring. Colour
    lerp(Color, metric, Intensity) with the metric normalised over the measured leaves (area, width, height; none =
    1), then Color Random (hue, saturation, lightness, alpha towards random values; gray: the lightness); random mask. Add/Sub draws (colour - Background
    Color) x pattern onto the background (a lerp to the colour), Max the maximum, Alpha Blend in order. With Non
    Square Expansion every square of the frame is a layout of its own seed. Leaves cross the tiled edges. Colour with a
    colour input or Color Variant."""

    type = "tile_random"
    main_input = "background_input"  # the library graph's primary input: Relative to input follows it
    label = "Tile Random"
    catalog = "tile_random"
    unsupported = {"Color Parametrisation Effect": "the documentation's name of Color Parametrisation Affect "
                                                   "(the node's)"}
    pattern_ports = ("pattern_input",)
    inputs = {"pattern_input": Input(PortType.ANY, "Pattern Input"),
              "background_input": Input(PortType.ANY, "Background Input")}
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {
        "x_amount": Param(I, 4, "X Amount", 1, 64),
        "y_amount": Param(I, 4, "Y Amount", 1, 64),
        **nse(),
        **color_variant(),
        # Pattern
        "pattern": Param(E, "Square", "Pattern", choices=pattern_choices("Pattern Input"), group="Pattern"),
        "image_input_filtering": Param(E, FILTERING[0], "Image Input Filtering", choices=FILTERING, group="Pattern"),
        "pattern_specific": Param(F, 0.0, "Pattern Specific", 0.0, 1.0, group="Pattern"),
        "pattern_specific_random": Param(F, 0.0, "Pattern Specific Random", 0.0, 1.0, group="Pattern"),
        "pattern_rotation": Param(E, "0", "Rotation", choices=ROTATIONS, group="Pattern"),
        "pattern_rotation_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Pattern"),
        "symmetry_random": Param(F, 0.0, "Symmetry Random", 0.0, 1.0, group="Pattern"),
        "symmetry_random_mode": Param(E, SYMMETRY[0], "Symmetry Random Mode", choices=SYMMETRY, group="Pattern"),
        # Split
        "split_mode": Param(E, "auto", "Mode", choices=SPLIT, group="Split"),
        "threshold": Param(F, 0.5, "Threshold", 0.0, 1.0, group="Split"),
        "multiplier": Param(I, 0, "Multiplier", 0, 10, group="Split"),
        # Size
        "random_x": Param(F, 0.5, "Random X", 0.0, 1.0, group="Size"),
        "random_y": Param(F, 0.5, "Random Y", 0.0, 1.0, group="Size"),
        # Interstice
        "interstice_mode": Param(E, INTERSTICE[1], "Mode", choices=INTERSTICE, group="Interstice"),
        "interstice_amount": Param(F, 0.2, "Amount", 0.0, 1.0, group="Interstice"),
        # Shape
        "scale": Param(F, 1.0, "Scale", 0.0, 1.0, group="Shape"),
        "scale_random": Param(F, 0.0, "Scale Random", 0.0, 1.0, group="Shape"),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0, group="Shape"),
        "rotation_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Shape"),
        "rotation_constraint": Param(B, True, "Rotation Constraint", group="Shape"),
        # Position
        "offset": Param(F, 0.0, "Offset", 0.0, 1.0, group="Position"),
        "offset_random": Param(F, 0.0, "Offset Random", 0.0, 1.0, group="Position"),
        "position_random": Param(F, 0.0, "Random", 0.0, 1.0, group="Position"),
        "random_constraints": Param(B, True, "Random Constraints", group="Position"),
        # Color
        "color": Param(C, [1.0, 1.0, 1.0, 1.0], "Color", group="Color"),
        # (hue, saturation, lightness, alpha); gray: the lightness only
        "color_random": Param(F4, [0.0, 0.0, 0.0, 0.0], "Color Random", 0.0, 1.0, group="Color"),
        "color_parametrisation": Param(E, "none", "Color Parametrisation", choices=PARAMETRISATION, group="Color"),
        "color_parametrisation_intensity": Param(F, 0.0, "Color Parametrisation Intensity", 0.0, 1.0, group="Color"),
        "color_parametrisation_affect": Param(E, AFFECT[1], "Color Parametrisation Affect", choices=AFFECT,
                                              group="Color"),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color", group="Color"),
        "blending_mode": Param(E, BLENDING[0], "Blending Mode", choices=BLENDING, group="Color"),
        # Mask
        "mask_random": Param(F, 0.0, "Random", 0.0, 1.0, group="Mask"),
        "mask_invert": Param(B, False, "Invert", group="Mask"),
    }

    def encode(self, ctx, job):
        data = layout_texture(job.values, job.size, job.seed, job.tiling, job.is_gray("output"))
        ctx.dispatch(DRAW, job, job.values, srcs={"leaves": (ctx.upload(lambda: data), False)})


class _Draw(TileRandom):
    """The pass drawing the uploaded leaves (tile_random.wgsl)."""
    type = "tile_random_draw"
    registered = False
    inputs = {**TileRandom.inputs, "leaves": Input(PortType.COLOR)}


DRAW = _Draw()


# --- layout (ours, per square of the frame) -------------------------------------------------------------------------

DEPTH = 2  # how many times a piece may be cut (fitted on the output)


def _pieces(rng, n: int, start: float, length: float, jitter: float) -> list[tuple[float, float]]:
    """(centre, size) of the n pieces of a span: a regular grid whose inner boundaries move by up to jitter / 2 of
    a piece, the ends fixed."""
    inner = (np.arange(1, n) + (rng.random(n - 1) - 0.5) * jitter) / n
    edges = np.concatenate([[0.0], inner, [1.0]]) * length
    return [(start + (a + b) / 2, b - a) for a, b in zip(edges[:-1], edges[1:])]


def _cut_count(mode: int, w: float, h: float, limit: float, extra: int, rng) -> tuple[int, int]:
    """(axis, count) of the cut of a w x h piece: across its long side (Split Mode may allow one axis only, or pick
    one at random) into ceil(elongation) near-square pieces when it is longer than `limit` times its width, plus
    `extra` (Multiplier, first cut only)."""
    if mode == 4:
        axis = int(rng.random() < 0.5)
    else:
        axis = int(h > w)
    long_, short = (w, h) if axis == 0 else (h, w)
    allowed = mode in (1, 4) or (mode == 2 and axis == 0) or (mode == 3 and axis == 1)
    if not allowed:
        return axis, 1
    elongation = long_ / max(short, 1e-6)
    n = ceil(elongation) if elongation > limit else 1
    return axis, n + extra


def _cut(rng, rect, axis: int, n: int, jitter: tuple[float, float]) -> list[tuple]:
    cx, cy, w, h = rect
    if axis == 0:
        return [(c, cy, s, h) for c, s in _pieces(rng, n, cx - w / 2, w, jitter[0])]
    return [(cx, c, w, s) for c, s in _pieces(rng, n, cy - h / 2, h, jitter[1])]


def _split(rng, rect, mode, limits, extra, jitter, depth) -> list[tuple]:
    """Recursive cuts of a piece: the first cut with limits[0] and Multiplier, the further ones with limits[1], at
    most `depth` levels."""
    axis, n = _cut_count(mode, rect[2], rect[3], limits[0], extra, rng)
    if n <= 1 or depth == 0:
        return [rect]
    out = []
    for piece in _cut(rng, rect, axis, n, jitter):
        # random h+v: the next cuts stay on the first cut's axis (its thin strips are not cut lengthwise)
        child = (2 if axis == 0 else 3) if mode == 4 else mode
        out += _split(rng, piece, child, (limits[1], limits[1]), 0, jitter, depth - 1)
    return out


def layout(v: dict, rng) -> np.ndarray:
    """The leaves (centre x, y, width, height in uv) of one square, in drawing order: Y Amount rows of X Amount
    bricks on a jittered grid, each row shifted by Offset (a staircase) and Offset Random, each brick cut by
    `_split`: a brick longer than 1 / (2 Threshold) times its width is cut; Threshold's upper half lets the pieces
    be cut again (longer than 1 / (2 Threshold - 1) times their width)."""
    nx, ny = max(v["x_amount"], 1), max(v["y_amount"], 1)
    jitter = (v["random_x"], v["random_y"])
    mode = SPLIT.index(v["split_mode"])
    thr = v["threshold"]
    limits = (1.0 / max(2.0 * thr, 1e-4), 1.0 / max(2.0 * thr - 1.0, 1e-4))
    leaves = []
    for j, (cy, bh) in enumerate(_pieces(rng, ny, 0.0, 1.0, jitter[1])):
        shift = v["offset"] * j / nx + rng.random() * v["offset_random"]
        for cx, bw in _pieces(rng, nx, 0.0, 1.0, jitter[0]):
            leaves += _split(rng, (cx + shift, cy, bw, bh), mode, limits, v["multiplier"], jitter, DEPTH)
    return np.array(leaves, np.float64).reshape(-1, 4)


def _discrete(u, a, b, probability):
    """A random choice per leaf: b with the given probability, else a."""
    return np.where(u < probability, b, a)


def leaf_rows(v: dict, leaves: np.ndarray, rng, gray: bool) -> np.ndarray:
    """Per drawn leaf (centre x, y, signed size x, y, rotation in turns, pattern specific, 0, 0, colour rgba): the
    sizes, moves, rotations, mirrors and colours of the drawing pass, from the leaves measured by the first."""
    n = len(leaves)
    u = rng.random((n, 16))
    w, h = leaves[:, 2], leaves[:, 3]
    area = w * h
    # The smallest and largest leaves (for the interstice and the colour metrics).
    lo, hi = w.min(initial=1.0), w.max(initial=0.0)
    lo_y, hi_y = h.min(initial=1.0), h.max(initial=0.0)
    lo_a, hi_a = area.min(initial=1.0), area.max(initial=0.0)
    # Pattern step (quarter turns): a fixed or random one; the odd ones swap the sizes.
    mode = ROTATIONS.index(v["pattern_rotation"])
    if mode >= 4:
        step = _discrete(u[:, 0], 0, 2, 0.5) + (mode == 5)
    else:
        step = np.where(u[:, 0] < v["pattern_rotation_random"], np.floor(u[:, 1] * 4.0), mode)
    turn = v["rotation"] + (u[:, 2] * 2.0 - 1.0) * v["rotation_random"]
    k = v["scale"] + (u[:, 3] - v["scale"]) * v["scale_random"]
    a = v["interstice_amount"]
    gap = a * a * (min(lo, lo_y) if INTERSTICE.index(v["interstice_mode"]) == 0 else min(hi, hi_y))
    size = np.stack([np.maximum(w - gap, 0.0), np.maximum(h - gap, 0.0)], 1) * k[:, None]
    fitted = size
    if v["rotation_constraint"]:  # the largest scale at which the turned tile still fits its own box
        t = turn * 2.0 * np.pi
        c, s = np.abs(np.cos(t)), np.abs(np.sin(t))
        bw = size[:, 0] * c + size[:, 1] * s
        bh = size[:, 0] * s + size[:, 1] * c
        f = np.minimum(size[:, 0] / np.maximum(bw, 1e-6), size[:, 1] / np.maximum(bh, 1e-6))
        fitted = size * np.minimum(f, 1.0)[:, None]
    fitted = np.where((step % 2 == 1)[:, None], fitted[:, ::-1], fitted)
    sym = SYMMETRY.index(v["symmetry_random_mode"])
    p_sym = v["symmetry_random"] * 0.5
    mirror = np.stack([_discrete(u[:, 4], 1.0, -1.0, p_sym) if sym != 2 else np.ones(n),
                       _discrete(u[:, 5], 1.0, -1.0, p_sym) if sym != 1 else np.ones(n)], 1)
    sign = np.stack([_discrete(u[:, 6], 0.5, -0.5, 0.5), _discrete(u[:, 7], 0.5, -0.5, 0.5)], 1)
    if v["random_constraints"]:  # pushed by half the free space in the leaf, never out of it
        move = v["position_random"] * (leaves[:, 2:] - size) * sign
    else:
        move = sign * (u[:, 8] * v["position_random"])[:, None]
    spec = v["pattern_specific"] + (u[:, 9] - v["pattern_specific"]) * v["pattern_specific_random"]
    # Colour: lerp(Color, metric, intensity), the lightness lerped to a random value, minus the background in Add/Sub.
    pm = PARAMETRISATION.index(v["color_parametrisation"])
    if pm == 1:
        d = area - lo_a
        metric = np.where(d == 0.0, k, d / max(hi_a - lo_a, 1e-4))
    elif pm == 2:
        metric = (w - lo) / max(hi - lo, 1e-4)
    elif pm == 3:
        metric = (h - lo_y) / max(hi_y - lo_y, 1e-4)
    else:
        metric = np.ones(n)
    color = np.array(v["color"], np.float64)
    bg = np.array(v["background_color"], np.float64)
    if gray:
        color, bg = np.array([color[0]] * 3 + [1.0]), np.array([bg[0]] * 3 + [1.0])
    affect = AFFECT.index(v["color_parametrisation_affect"]) if not gray else 0
    target = np.repeat(metric[:, None], 4, 1)
    if affect == 1:
        target[:, 3] = color[3]
    elif affect == 2:
        target[:, :3] = color[:3]
    rgba = color + (target - color) * v["color_parametrisation_intensity"]
    cr = v["color_random"]
    rgba = _color_random(rgba, u[:, [12, 13, 10, 14]], (0.0, 0.0, cr[2], 0.0) if gray else cr)
    if BLENDING.index(v["blending_mode"]) == 0:
        rgba[:, :3] -= bg[:3]
    shown = u[:, 11] > v["mask_random"]
    if v["mask_invert"]:
        shown = ~shown
    rows = np.zeros((n, 12))
    rows[:, 0:2] = leaves[:, :2] + move
    rows[:, 2:4] = fitted * mirror
    rows[:, 4] = turn + 0.25 * step
    rows[:, 5] = spec
    rows[:, 8:12] = rgba
    return rows[shown]


def _color_random(rgba: np.ndarray, u: np.ndarray, r) -> np.ndarray:
    """Color Random per leaf (lib.wgsl `color_random`): the hue shifted by r[0] x (u0 - 0.5), saturation and
    lightness lerped to u1, u2 by r[1], r[2], alpha to u3 x alpha by r[3] (grayscale: lightness only)."""
    if not any(r):
        return rgba
    rgb = np.clip(rgba[:, :3], 0.0, 1.0)
    hi, lo = rgb.max(1), rgb.min(1)
    lit, d = (hi + lo) / 2.0, hi - lo
    sat = np.where(d > 1e-6, d / np.maximum(1.0 - np.abs(2.0 * lit - 1.0), 1e-6), 0.0)
    rr, gg, bb = rgb.T
    hue = np.where(hi == rr, (gg - bb) / np.maximum(d, 1e-6),
                   np.where(hi == gg, (bb - rr) / np.maximum(d, 1e-6) + 2.0, (rr - gg) / np.maximum(d, 1e-6) + 4.0))
    hue = np.mod(np.where(d > 1e-6, hue / 6.0, 0.0) + r[0] * (u[:, 0] - 0.5), 1.0)
    sat = sat + (u[:, 1] - sat) * r[1]
    lit = lit + (u[:, 2] - lit) * r[2]
    k = np.mod(hue[:, None] + np.array([0.0, 2.0, 1.0]) / 3.0, 1.0) * 6.0
    t = np.clip(np.abs(k - 3.0) - 1.0, 0.0, 1.0)
    out = rgba.copy()
    out[:, :3] = lit[:, None] + (sat * (1.0 - np.abs(2.0 * lit - 1.0)))[:, None] * (t - 0.5)
    out[:, 3] = rgba[:, 3] + (u[:, 3] * rgba[:, 3] - rgba[:, 3]) * r[3]
    return out


def layout_texture(v: dict, size, seed: float, tiling: int, gray: bool) -> np.ndarray:
    """The leaves of every square of the frame (each its own seed, with Non Square Expansion) with
    their copies across the tiled edges, binned on a grid over the frame, packed as rgba32 texels (DATA_WIDTH per
    row): texel 0 = (bins x, bins y, first entry, first leaf), then per bin (first entry, count), then per entry its
    leaf, then 3 texels per leaf (the rows of `leaf_rows`)."""
    frame = np.array(size, np.float64) / min(size) if v["non_square_expansion"] else np.ones(2)
    squares = [(x, y) for y in range(int(np.ceil(frame[1]))) for x in range(int(np.ceil(frame[0])))]
    bits = struct.unpack("<I", struct.pack("<f", seed))[0]
    rows = []
    for i, (x, y) in enumerate(squares):
        r = leaf_rows(v, layout(v, np.random.default_rng([bits, i, 0])), np.random.default_rng([bits, i, 1]), gray)
        r[:, 0:2] += (x, y)
        rows.append(r)
    rows = np.concatenate(rows)
    # Copies across the tiled edges (one frame away).
    shifts = [(sx, sy) for sx in ([0.0, -frame[0], frame[0]] if tiling & 1 else [0.0])
              for sy in ([0.0, -frame[1], frame[1]] if tiling & 2 else [0.0])]
    rows = np.repeat(rows, len(shifts), 0)
    rows[:, 0:2] += np.tile(np.array(shifts), (len(rows) // len(shifts), 1))
    # Extent of each rotated pattern; the leaves outside the frame are dropped.
    t = rows[:, 4] * 2.0 * np.pi
    hx, hy = np.abs(rows[:, 2]) / 2, np.abs(rows[:, 3]) / 2
    ex = np.abs(np.cos(t)) * hx + np.abs(np.sin(t)) * hy
    ey = np.abs(np.sin(t)) * hx + np.abs(np.cos(t)) * hy
    lo = np.stack([rows[:, 0] - ex, rows[:, 1] - ey], 1)
    hi = np.stack([rows[:, 0] + ex, rows[:, 1] + ey], 1)
    keep = np.all(hi >= 0.0, 1) & np.all(lo <= frame, 1) & (hx > 0) & (hy > 0)
    rows, lo, hi = rows[keep], lo[keep], hi[keep]
    # Bins: about one per brick.
    per = max(v["x_amount"], v["y_amount"], 1)
    grid = np.maximum(np.ceil(frame * per).astype(int), 1)
    a = np.clip(np.floor(lo / frame * grid).astype(int), 0, grid - 1)
    b = np.clip(np.floor(hi / frame * grid).astype(int), 0, grid - 1)
    span = b - a + 1
    count = span[:, 0] * span[:, 1]
    leaf = np.repeat(np.arange(len(rows)), count)
    local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    cell = (a[leaf, 1] + local // span[leaf, 0]) * grid[0] + a[leaf, 0] + local % span[leaf, 0]
    order = np.lexsort((leaf, cell))
    entries, cells = leaf[order], cell[order]
    counts = np.bincount(cells, minlength=grid[0] * grid[1])
    first_entry = 1 + len(counts)
    first_leaf = first_entry + len(entries)
    data = np.zeros((first_leaf + 3 * len(rows), 4), np.float32)
    data[0] = (grid[0], grid[1], first_entry, first_leaf)
    data[1:first_entry, 0] = np.cumsum(counts) - counts
    data[1:first_entry, 1] = counts
    data[first_entry:first_leaf, 0] = entries
    data[first_leaf:] = rows.reshape(-1, 4)
    pad = -len(data) % DATA_WIDTH
    return np.concatenate([data, np.zeros((pad, 4), np.float32)]).reshape(-1, DATA_WIDTH, 4)
