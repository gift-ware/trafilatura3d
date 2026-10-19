"""The machinery the Legacy generators share: the node interface, FX-Map quad-trees and helpers over the atomic nodes.

Node interface (`LegacyNode`): `image` draws the node with values `v` at `seed` at the job's size, into `dst` (a new
gray scratch, a texture, or "output": the job's) and returns it as (texture, gray); `encode` draws the node's own
values into its output, `instance` draws it as a part of another legacy node (its values over the defaults, its seed,
at a size) and returns the same image the node draws. Images flow as (texture, gray) pairs of whatever size the step
that made them ran at.

FX-Maps are full quad-trees generated in numpy (a Quadrant at depth d has 2^d x 2^d cells, its Branch Offset in uv
carried by all its descendants, its Frame Offset only its own pattern's, its Pattern Size in cells; Roughness and
Global opacity weigh the tree's levels) and drawn by the FX-Map engine (atomic/stamp_map.py `draw_patterns`); random
values come from a numpy stream per node type and seed (`stream`).

Each step runs at its own size, relative or absolute: a step reading inputs of another size gets them resampled
bilinearly (`fit`, averaged when smaller). Nodes with absolute sizes say so (`LegacyNode.absolute_sizes`): their look
depends on the output's resolution."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np

from trafilatura3d.fxmap import BLENDING, PATTERNS
from trafilatura3d.nodes.atomic.blend import Blend
from trafilatura3d.nodes.atomic.blur import Blur
from trafilatura3d.nodes.atomic.directional_blur import directional_blur
from trafilatura3d.nodes.atomic.directional_warp import DirectionalWarp
from trafilatura3d.nodes.atomic.grayscale_conversion import GrayscaleConversion
from trafilatura3d.nodes.atomic.levels import Levels
from trafilatura3d.nodes.atomic.normal import Normal
from trafilatura3d.nodes.atomic.sharpen import Sharpen
from trafilatura3d.nodes.atomic.stamp_map import ROW, FxMap, draw_patterns
from trafilatura3d.nodes.atomic.transformation_2d import Transformation2D
from trafilatura3d.nodes.atomic.warp import Warp
from trafilatura3d.nodes.base import Node, Output, Param, ParamKind, PortType

from ..noises_a.fractal_sum import FractalSumBase
from ..noises_b.noises import WhiteNoise

CATEGORY = "Texture generators/Legacy"
GRAY_OUT = {"output": Output(PortType.GRAY, "Output")}
F, I, B, E = ParamKind.FLOAT, ParamKind.INT, ParamKind.BOOL, ParamKind.ENUM
F2, F4 = ParamKind.FLOAT2, ParamKind.FLOAT4
DISORDER = {"disorder": Param(F, 0.0, "Disorder", 0.0, 100.0)}

LEVELS, TRANSFORM, BLEND, SHARPEN, BLUR = Levels(), Transformation2D(), Blend(), Sharpen(), Blur()
WARP, DWARP, NORMAL, GRAY = Warp(), DirectionalWarp(), Normal(), GrayscaleConversion()
WHITE, FRACTAL = WhiteNoise(), FractalSumBase()
ADD_SUB, MAX = BLENDING.index("Add Sub"), BLENDING.index("Max")
TILE_2, IDENTITY, TURN_90 = (2.0, 0.0, 0.0, 2.0), (1.0, 0.0, 0.0, 1.0), (0.0, -1.0, 1.0, 0.0)


def stream(seed: float, node: Node) -> np.random.Generator:
    """The random stream of a legacy node at `seed`."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, node.stream_salt()])


def fx_values(background: float, opacity: float, roughness: float = 1.0) -> dict:
    """The FX-Map node's values: the patterns' layer added over a gray Background, times Global opacity."""
    return {**{k: p.default for k, p in FxMap.params.items()},
            "background": [background, background, background, 1.0], "global_opacity": opacity,
            "roughness": roughness}


# --- quad-trees ------------------------------------------------------------------------------------------------------

def cells(depth: int) -> np.ndarray:
    """The centres of the 2^depth x 2^depth cells of a full quad-tree level, (n, n, 2) indexed [y, x]."""
    n = 1 << depth
    gy, gx = np.mgrid[0:n, 0:n]
    return np.stack([(gx + 0.5) / n, (gy + 0.5) / n], -1)


def inherit(offset: np.ndarray) -> np.ndarray:
    """A level's Branch Offsets (n, n, 2) handed to the next level's four children of each cell."""
    return offset.repeat(2, 0).repeat(2, 1)


def stamps(pos: np.ndarray, size: np.ndarray, pattern: str, lum: np.ndarray, depth: int, blend: int = ADD_SUB,
           rotation: np.ndarray | float = 0.0, variation: np.ndarray | float = 0.0) -> np.ndarray:
    """FX-Map rows (atomic/stamp_map.py `rows_of`) of patterns at `pos` (k, 2) in uv, `size` (k, 2) in uv, luminosity
    `lum` (k,), rotation in turns; positions as the offsets took them (the engine wraps them along the tiled axes
    only: a branch moved off an untiled side is not drawn, as Dirt gradient's)."""
    rows = np.zeros((len(pos), ROW), np.float32)
    rows[:, 0:2] = pos
    rows[:, 2:4] = size
    rows[:, 4] = rotation
    rows[:, 5] = variation
    rows[:, 6] = PATTERNS.index(pattern)
    rows[:, 8:11] = lum[:, None]
    rows[:, 11] = 1.0
    rows[:, 14] = blend
    rows[:, 15] = depth
    return rows


Fn = Callable[[np.random.Generator, int], np.ndarray]  # a quadrant's function: k values for k quadrants


def rand(a: float) -> Fn:
    """Uniform in 0..a, one per quadrant."""
    return lambda rng, k: a * rng.random(k)


def polar(rho: float | Fn, spread: float = 6.28) -> Fn:
    """An offset of length rho (a value or a function) at a random angle in 0..spread."""
    def f(rng, k):
        r = rho(rng, k) if callable(rho) else np.full(k, rho)
        theta = rng.random(k) * spread
        return np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
    return f


def angles(rho: float | Fn, theta: float = 6.28) -> Fn:
    """An offset (rho cos(a), rho sin(b)) with a separate random angle per component (a, b uniform in 0..theta)."""
    def f(rng, k):
        r = rho(rng, k) if callable(rho) else np.full(k, float(rho))
        return np.stack([r * np.cos(rng.random(k) * theta), r * np.sin(rng.random(k) * theta)], 1)
    return f





@dataclass(frozen=True)
class Level:
    """The Quadrants of one depth of a legacy FX-Map (all its quadrants are the same node)."""
    offset: Fn | None = None          # Branch Offset (k, 2) in uv
    pattern: str = ""                 # fxmap.PATTERNS; "" draws nothing
    size: Fn | tuple = (1.0, 1.0)     # Pattern Size (k, 2) in cells
    lum: Fn | float = 1.0             # Luminosity (with Add Sub: signed)
    rotation: Fn | float = 0.0        # in turns
    frame: Fn | None = None           # Frame (pattern) offset (k, 2) in uv
    children: tuple = (0, 1, 2, 3)    # the slots each quadrant connects to the next depth (0 top left .. 3 bottom right)


def _values(x, rng, k, width):
    if callable(x):
        return x(rng, k)
    return np.tile(np.asarray(x, np.float64), (k, 1)) if width == 2 else np.full(k, float(x))


def quadtree(levels_: list[Level], rng: np.random.Generator) -> np.ndarray:
    """The FX-Map rows of a quad-tree whose depth d is levels_[d], from the root (one quadrant) down."""
    centre, offset, out = np.array([[0.5, 0.5]]), np.zeros((1, 2)), []
    for depth, q in enumerate(levels_):
        if depth:
            corner = np.array([((s % 2) - 0.5, (s // 2) - 0.5) for s in levels_[depth - 1].children]) / (1 << depth)
            centre = (centre[:, None] + corner[None]).reshape(-1, 2)
            offset = np.repeat(offset, len(corner), 0)
        k = len(centre)
        if q.offset is not None:
            offset = offset + q.offset(rng, k)
        if q.pattern:
            pos = centre + offset + (q.frame(rng, k) if q.frame is not None else 0.0)
            out.append(stamps(pos, _values(q.size, rng, k, 2) / (1 << depth), q.pattern, _values(q.lum, rng, k, 1),
                              depth, rotation=_values(q.rotation, rng, k, 1)))
    return np.concatenate(out)


# --- atomic steps at their sizes; an image is (texture, gray) --------------------------------------------------------

def dims(img) -> tuple[int, int]:
    return img[0].width, img[0].height


def rel(job, kx: int, ky: int | None = None) -> tuple[int, int]:
    """A size relative to the job's: 2^kx x 2^ky (ky default kx) times it."""
    ky = kx if ky is None else ky
    return max(1, int(job.size[0] * 2.0 ** kx)), max(1, int(job.size[1] * 2.0 ** ky))


def sized(job, size: tuple[int, int]):
    """The job as the atomic nodes see it when a node runs at `size` (same key: "output" is still the job's)."""
    return job if tuple(size) == tuple(job.size) else dataclasses.replace(job, size=tuple(size))


def target(ctx, job, dst):
    """Where a node writes: `dst`, the job's output ("output") or a new gray scratch at the job's size (None)."""
    if dst is None:
        return ctx.scratch(job, True, "32f")
    return ctx.ev.texture(job, "output") if isinstance(dst, str) else dst


def apply(ctx, job, node: Node, srcs: dict, dst=None, **values):
    """One pass of an atomic node (its defaults and `values`) on `srcs` (port -> image) at the job's size, into
    `target(dst)`; returns the image."""
    out = target(ctx, job, dst)
    ctx.dispatch(node, job, {**{k: p.default for k, p in node.params.items()}, **values}, srcs=srcs,
                 dsts={"output": out})
    return out, True


def fit(ctx, job, img, size):
    """`img` at `size`: resampled bilinearly (averaged when smaller) if it is another size."""
    return img if dims(img) == tuple(size) else xf(ctx, job, img, IDENTITY, size)


def draw(ctx, job, rows: np.ndarray, values: dict, size=None, levels: int | None = None, clamp: bool = False,
         tiling: int | None = None, images: list | None = None, dst=None, cull=None):
    """FX-Map rows drawn at `size` (default: the job's) with the FX-Map node's `values`; `levels`: the tree's (default:
    the deepest row's), `clamp` as an 8 or 16 bit FX-Map, `tiling` the FX-Map's own (default the job's); `cull`:
    (texture, threshold), the rows whose centre reads less there are left out."""
    j = sized(job, size or job.size)
    if tiling is not None:
        j = dataclasses.replace(j, tiling=tiling)
    out = target(ctx, j, dst)
    draw_patterns(ctx, j, rows, values, images, dst=out, levels=levels, clamp=clamp, cull=cull)
    return out, True


def fx(ctx, job, levels_: list[Level], rng, values: dict, size=None, tiling: int | None = None):
    """A legacy FX-Map quad-tree drawn at `size` (clamped, as its 16 bits)."""
    return draw(ctx, job, quadtree(levels_, rng), values, size, len(levels_), True, tiling)


def levels(ctx, job, img, low=0.0, high=1.0, mid=0.5, out_low=0.0, out_high=1.0, dst=None):
    """The atomic Levels (clamped) at the input's size."""
    return apply(ctx, sized(job, dims(img)), LEVELS, {"input": img}, dst,
                 level_in_low=[low] * 4, level_in_high=[high] * 4, level_in_mid=[mid] * 4,
                 level_out_low=[out_low] * 4, level_out_high=[out_high] * 4)


def xf(ctx, job, img, matrix, size=None, offset=(0.0, 0.0), dst=None):
    """A Transformation 2D (sampling matrix and offset) at `size` (default: the input's)."""
    return apply(ctx, sized(job, size or dims(img)), TRANSFORM, {"input": img}, dst,
                 transformation_matrix=list(matrix), offset=list(offset))


def mix(ctx, job, background, foreground, mode: str, opacity: float = 1.0, mask=None, size=None, dst=None):
    """The atomic Blend (destination = background, source = foreground, Opacity input = mask) at `size` (default:
    the background's); Add and Add Sub clamped."""
    size = size or dims(background)
    j = sized(job, size)
    srcs = {"background": fit(ctx, job, background, size), "foreground": fit(ctx, job, foreground, size)}
    if mask is not None:
        srcs["opacity"] = fit(ctx, job, mask, size)
    if mode in ("Add", "Add sub"):
        return levels(ctx, job, apply(ctx, j, BLEND, srcs, blending_mode=mode, opacity=opacity), dst=dst)
    return apply(ctx, j, BLEND, srcs, dst, blending_mode=mode, opacity=opacity)


def blur(ctx, job, img, intensity: float, size=None, pixel_size: float = 1.0):
    """The atomic Blur's two passes (pixels at 256 x Pixel Size) at `size` (default: the input's)."""
    j = sized(job, size or dims(img))
    rows = apply(ctx, j, BLUR, {"input": fit(ctx, job, img, j.size)}, intensity=intensity, vertical=0,
                 pixel_size_=(pixel_size, pixel_size))
    return apply(ctx, j, BLUR, {"input": rows}, intensity=intensity, vertical=1, pixel_size_=(pixel_size, pixel_size))


def motion_blur(ctx, job, img, intensity: float):
    """The atomic Directional Blur at angle 0 (horizontal) at the input's size."""
    j = sized(job, dims(img))
    return directional_blur(ctx, j, img, True, intensity, 0.0, ctx.scratch(j, True, "32f")), True


def warp(ctx, job, img, gradient, intensity: float, pixel_size: float = 1.0, dst=None):
    """The atomic Warp (Intensity / 256 x the gradient's slope in uv x Pixel Size) at the input's size, the gradient
    resampled to it."""
    size = dims(img)
    return apply(ctx, sized(job, size), WARP, {"input": img, "gradient_input": fit(ctx, job, gradient, size)}, dst,
                 intensity=intensity, pixel_size_=(pixel_size, pixel_size))


def white(ctx, job, size):
    """The White noise node (one uniform value per pixel) at `size`."""
    return apply(ctx, sized(job, size), WHITE, {})


def fractal_sum(ctx, job, size, **values):
    """The Fractal sum base node (`values` its parameters) at `size`."""
    return apply(ctx, sized(job, size), FRACTAL, {}, **values)


def output(ctx, job, img, dst, **values):
    """A last Levels (`values` as `levels`') at the job's size, into `dst`."""
    return levels(ctx, job, fit(ctx, job, img, job.size), dst=dst, **values)


# --- the node interface ----------------------------------------------------------------------------------------------

class LegacyNode(Node):
    """A legacy node: `image` draws it with values `v` at `seed` at the job's size into `dst` and returns it."""
    category = CATEGORY
    outputs = GRAY_OUT
    # Some steps run at absolute sizes: the look changes with the output's resolution.
    absolute_sizes: ClassVar[bool] = False

    def image(self, ctx, job, v: dict, seed: float, dst=None):
        raise NotImplementedError

    def encode(self, ctx, job):
        self.image(ctx, job, job.values, job.seed, "output")

    def instance(self, ctx, job, values: dict | None = None, seed: float | None = None, size=None, **inputs):
        """This node as a part of another one: its `values` over the defaults, at `seed` (default the job's), drawn at
        `size` (default the job's) into a scratch; `inputs`: its connected inputs as images (by port)."""
        v = {**{k: p.default for k, p in self.params.items()}, **(values or {})}
        return self.image(ctx, sized(job, size or job.size), v, job.seed if seed is None else seed, **inputs)


class LegacyNoise(LegacyNode):
    """One FX-Map (`rows`, drawn with `FX` at the job's size) then `finish`: by default a Levels
    (`LEVELS`)."""
    FX: ClassVar[dict]
    LEVELS: ClassVar[dict] = {}

    def rows(self, v: dict, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def finish(self, ctx, job, v: dict, seed: float, img, dst):
        return levels(ctx, job, img, dst=dst, **self.LEVELS)

    def image(self, ctx, job, v, seed, dst=None):
        return self.finish(ctx, job, v, seed, draw(ctx, job, self.rows(v, stream(seed, self)), self.FX), dst)
