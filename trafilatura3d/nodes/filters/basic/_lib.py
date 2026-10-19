"""Shared pieces of the basic filters: the WGSL helpers prepended to every shader of this folder
(`lib.wgsl`), helper nodes for passes that several filters need (1D blur, image statistics by
reduction, plain copy) and the Python side that records them."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, ParamKind, PortType

DIR = Path(__file__).parent
LIB = (DIR / "lib.wgsl").read_text(encoding="utf-8")


class BasicNode(Node):
    """A node of this folder: its shader gets the shared helpers. `defines` adds WGSL constants in
    front (for several node types sharing one shader file)."""

    defines: dict = {}

    def wgsl(self) -> str:
        consts = "".join(f"const {k} = {v};\n" for k, v in self.defines.items())
        return consts + LIB + "\n" + Path(inspect.getfile(type(self))).with_suffix(".wgsl").read_text(encoding="utf-8")


def helper(type_: str, shader: str | tuple[str, ...], inputs: dict, outputs: dict, params: dict | None = None,
           internal: dict | None = None, defines: dict | None = None) -> Node:
    """A pass with its own shader (`shader`: file name(s) in this folder, concatenated), not in the
    library."""
    files = (shader,) if isinstance(shader, str) else shader

    def wgsl(self):
        consts = "".join(f"const {k} = {v};\n" for k, v in (defines or {}).items())
        return consts + LIB + "\n" + "\n".join((DIR / f).read_text(encoding="utf-8") for f in files)

    cls = type(f"_{type_}", (Node,), {"type": type_, "registered": False, "inputs": inputs, "outputs": outputs,
                                      "params": params or {}, "internal_uniforms": internal or {}, "wgsl": wgsl})
    return cls()


def source(ctx, job, port: str):
    """(texture, is_gray) of what is connected to `port`, or of the input's default value."""
    s = job.inputs.get(port)
    if s:
        return ctx.ev.texture(s[0], s[1]), s[0].is_gray(s[1])
    spec = job.node.inputs[port]
    return ctx.ev.constant(spec.default), spec.type is PortType.GRAY


def source_size(job, port: str) -> tuple[int, int]:
    s = job.inputs.get(port)
    return s[0].size if s else (1, 1)


def scale(job) -> float:
    """Output pixels per pixel of a 256 px image (distances in the UI are 'pixels at 256')."""
    return job.size[0] / 256.0


# --- 1D blur along a direction (box or gaussian) ---------------------------------------------------

LINE = helper("basic_line", "line.wgsl", {"input": Input()}, {"output": Output()},
              internal={"direction": ParamKind.FLOAT2, "radius": ParamKind.FLOAT,
                        "kernel": ParamKind.INT, "taps": ParamKind.INT})
BOX, GAUSS = 0, 1


DOWN = helper("basic_down", "down.wgsl", {"input": Input()}, {"output": Output()},
              internal={"factor": ParamKind.INT2})
UP = {axis: helper(f"basic_up_{name}", "up.wgsl", {"input": Input()}, {"output": Output()},
                   defines={"AXIS": f"vec2i({axis[0]}, {axis[1]})"}) for name, axis in (("x", (1, 0)), ("y", (0, 1)))}
REDUCED_SIGMA = 2.0  # a blur runs on a copy reduced until its narrowest sigma is under twice this, in texels
# Variance (in texels of the reduced copy) that the box reduction and the enlargement add: the blur on the copy is
# narrowed by it, so the result keeps the width it has without the reduction. The copy is enlarged with the cubic
# B-spline of up.wgsl (`enlarge`), whose slope has no jumps.
BOX_SPREAD = 1.0 / 12.0
BSPLINE_SPREAD = 1.0 / 3.0


def line(ctx, job, src, dst, direction, radius_px: float, kernel: int = GAUSS, taps: int = 48, size=None):
    """One 1D blur pass of `src` ((texture, gray)) into `dst` (texture, or None = the job's output).
    `radius_px`: gaussian sigma or box half-width, in pixels of the output (`size`, default the job's)."""
    values = {"direction": [float(direction[0]), float(direction[1])], "radius": float(max(radius_px, 0.0)),
              "kernel": kernel, "taps": taps}
    ctx.dispatch(LINE, job, values, srcs={"input": src}, dsts={"output": dst} if dst is not None else None,
                 size=size)


def kept(tex) -> str:
    """The depth of a temporary holding what an average of `tex` with positive weights makes (a box reduction, a
    gaussian or box line, a B-spline enlargement): its values stay in the texture's range, so an integer texture's
    (0-1) go in 16 bit integers, exact to our 16 bit results at half a 32 bit float's memory traffic; a float
    texture's in its own depth."""
    fmt = tex.format
    return "16" if "unorm" in fmt else "16f" if "16float" in fmt else "32f"


def downsample(ctx, job, tex, gray: bool, factor: tuple[int, int]):
    """`tex` reduced by box averages of factor.x x factor.y texels (a partial block at the edges averages what it
    has); returns the new scratch."""
    w, h = tex.width, tex.height
    size = (max(1, -(-w // factor[0])), max(1, -(-h // factor[1])))
    out = ctx.scratch(job, gray, kept(tex), size=size)
    ctx.dispatch(DOWN, job, {"factor": list(factor)}, srcs={"input": (tex, gray)}, dsts={"output": out}, size=size)
    return out


def _sigma(radius: float, kernel: int) -> float:
    """A blur's width as a sigma: a box of half-width r is r / sqrt(3)."""
    return radius if kernel == GAUSS else radius / math.sqrt(3.0)


def reduced_radius(radius: float, f: int, kernel: int) -> float:
    """The radius, in texels of a copy reduced by f, that gives the blur `radius` (full-size texels) once the copy's
    reduction and enlargement are added (BOX_SPREAD, BSPLINE_SPREAD)."""
    if f == 1 or radius <= 0.0:
        return radius
    sigma = math.sqrt(max((_sigma(radius, kernel) / f) ** 2 - BOX_SPREAD - BSPLINE_SPREAD, 0.0))
    return sigma if kernel == GAUSS else sigma * math.sqrt(3.0)


def reduction(size: tuple[int, int], radii: list[float], kernel: int) -> int:
    """The power of two a blur's copy is reduced by: the narrowest blur (as a sigma: a box of half-width r is
    r / sqrt(3)) keeps at least REDUCED_SIGMA texels there, and the size divides exactly. A wide gaussian has no
    detail a reduced copy loses: the box prefilter and the enlargement add their variance, taken off the blur on the
    copy (`reduced_radius`). With 2 texels the cost of a blur peaks at a sigma
    of 4 output pixels and stays there or lower for every wider one."""
    sigmas = [_sigma(r, kernel) for r in radii if r > 0.0]
    if not sigmas:
        return 1
    w, h = size
    f = 1
    while min(sigmas) / (f * 2) >= REDUCED_SIGMA and w % (f * 2) == 0 and h % (f * 2) == 0:
        f *= 2
    return f


def gaussian(ctx, job, tex, gray: bool, radius: float, kernel: int = GAUSS, taps: int = 48, u=(1.0, 0.0),
             radius_v: float | None = None, dst=None, size: tuple[int, int] | None = None):
    """The blur of every node (the one path): separable along `u` then its perpendicular (radius_v, default the
    same), radii in texels of `tex`, into `dst` (default a new scratch) of `size` (default the texture's). Wide blurs
    run on a copy reduced by `reduction`, then enlarged (`enlarge`): their cost stays about the same at any radius
    and resolution. The job's tiling says whether it wraps at the borders."""
    rv = radius if radius_v is None else radius_v
    w, h = tex.width, tex.height
    target = tuple(size) if size is not None else (w, h)
    depth = kept(tex)
    out = dst if dst is not None else ctx.scratch(job, gray, depth, size=target)
    f = reduction((w, h), [radius, rv], kernel)
    size = (w // f, h // f)
    src, left = (tex, gray), f
    while left > 1:  # by at most 16 a pass: a single pass by 32 gave each thread 1024 texels to read in a row
        step = min(left, 16)
        src, left = (downsample(ctx, job, src[0], gray, (step, step)), gray), left // step
    a = ctx.scratch(job, gray, depth, size=size)
    b = out if size == target else ctx.scratch(job, gray, depth, size=size)
    line(ctx, job, src, a, u, reduced_radius(radius, f, kernel), kernel, taps, size=size)
    line(ctx, job, (a, gray), b, (-u[1], u[0]), reduced_radius(rv, f, kernel), kernel, taps, size=size)
    if b is not out:
        enlarge(ctx, job, (b, gray), out)
    return out


def blur2d(ctx, job, src, gray: bool, radius_px: float, kernel: int = GAUSS, taps: int = 48,
           dst=None, u=(1.0, 0.0), radius_v: float | None = None):
    """`gaussian` of `src` ((texture, gray)) at the job's size, radii in output pixels, into `dst` (default a new
    scratch); returns the texture written."""
    tex = src[0]
    k = tex.width / job.size[0]  # output pixels -> texels of the source
    rv = radius_px if radius_v is None else radius_v
    return gaussian(ctx, job, tex, gray, radius_px * k, kernel, taps, u, rv * k, dst=dst, size=job.size)


def enlarge(ctx, job, src, dst):
    """`src` ((texture, gray)) resampled to the size of `dst` with a cubic B-spline (up.wgsl: no creases along the
    source's texels, BSPLINE_SPREAD of variance), one pass per axis whose size changes, the smaller in-between
    first."""
    (sw, sh), (dw, dh), gray = (src[0].width, src[0].height), (dst.width, dst.height), src[1]
    passes = [(axis, size) for axis, size, changes in (((0, 1), (sw, dh), sh != dh), ((1, 0), (dw, sh), sw != dw))
              if changes]
    passes.sort(key=lambda a: a[1][0] * a[1][1])
    for k, (axis, size) in enumerate(passes):
        last = k == len(passes) - 1
        size = (dw, dh) if last else size
        out = dst if last else ctx.scratch(job, gray, kept(src[0]), size=size)
        ctx.dispatch(UP[axis], job, {}, srcs={"input": src}, dsts={"output": out}, size=size)
        src = (out, gray)


# --- image statistics: min, max, mean by reduction ----------------------------------------------

_STATS = {"lo": Output(PortType.COLOR), "hi": Output(PortType.COLOR), "mean": Output(PortType.COLOR)}
REDUCE_FIRST = helper("basic_reduce_first", "reduce_first.wgsl", {"input": Input()}, _STATS,
                      internal={"factor": ParamKind.INT2, "kind": ParamKind.INT})
REDUCE_NEXT = helper("basic_reduce_next", "reduce_next.wgsl",
                     {"lo": Input(PortType.COLOR), "hi": Input(PortType.COLOR), "mean": Input(PortType.COLOR)},
                     _STATS, internal={"factor": ParamKind.INT2, "kind": ParamKind.INT})
VALUES, BOX_RGB, BOX_ALPHA = 0, 1, 2


def reduce(ctx, job, src, src_size, kind: int = VALUES):
    """Statistics of a whole image on the GPU, as three 1x1 rgba32float textures (lo, hi, mean).
    kind VALUES: per-channel min, max, mean; BOX_RGB / BOX_ALPHA: bounding box of the pixels whose
    colour (or alpha) is above 0, lo = (x0, y0), hi = (x1, y1) in pixels of the source."""
    w, h = src_size
    first = True
    while True:
        nw, nh = max(1, w // 8), max(1, h // 8)
        outs = {k: ctx.scratch(job, False, "32f", size=(nw, nh)) for k in _STATS}
        values = {"factor": [w // nw, h // nh], "kind": kind}
        if first:
            ctx.dispatch(REDUCE_FIRST, job, values, srcs={"input": src}, dsts=outs, size=(nw, nh))
        else:
            ctx.dispatch(REDUCE_NEXT, job, values, srcs={k: (prev[k], False) for k in _STATS}, dsts=outs,
                         size=(nw, nh))
        prev, first, w, h = outs, False, nw, nh
        if (w, h) == (1, 1):
            return prev["lo"], prev["hi"], prev["mean"]


# --- copy (switches) -------------------------------------------------------------------------------

COPY = helper("basic_copy", "copy.wgsl", {"input": Input()}, {"output": Output()})
