"""Shared pieces of the effect and normal map filters: the WGSL helpers prepended to every shader of
this folder (`_lib.wgsl`, plus per-node extra files in `libs`), helper passes several nodes need
(jump flooding, gaussian blur, prefix sums, min/max reduction) and the Python that records them."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, ParamKind, PortType

DIR = Path(__file__).parent


def _read(name: str) -> str:
    return (DIR / name).read_text(encoding="utf-8")


class Effect(Node):
    """A node of this folder: its shader is `_lib.wgsl`, the files in `libs`, then its own `.wgsl`."""

    libs: tuple[str, ...] = ()

    def wgsl(self) -> str:
        own = Path(inspect.getfile(type(self))).with_suffix(".wgsl").read_text(encoding="utf-8")
        return "\n".join([_read("_lib.wgsl"), *(_read(f) for f in self.libs), own])


def helper(type_: str, shader: str, inputs: dict, outputs: dict, internal: dict | None = None,
           params: dict | None = None, libs: tuple[str, ...] = (), buffers: dict | None = None) -> Node:
    """A pass with its own shader (a file of this folder), not in the library."""

    def wgsl(self):
        return "\n".join([_read("_lib.wgsl"), *(_read(f) for f in libs), _read(shader)])

    cls = type(f"_{type_}", (Node,), {"type": type_, "registered": False, "inputs": inputs, "outputs": outputs,
                                      "params": params or {}, "internal_uniforms": internal or {},
                                      "buffers": buffers or {}, "wgsl": wgsl})
    return cls()


def source(ctx, job, port: str):
    """(texture, is_gray) of what is connected to `port`, or of the input's default value."""
    s = job.inputs.get(port)
    if s:
        return ctx.ev.texture(s[0], s[1]), s[0].is_gray(s[1])
    spec = job.node.inputs[port]
    return ctx.ev.constant(spec.default), spec.type is PortType.GRAY


def scale(job) -> float:
    """Output pixels per pixel of a 256 px image (distances in the UI are 'pixels at 256')."""
    return job.size[0] / 256.0


# --- jump flooding: nearest white and nearest black pixel of a mask ---------------------------------

_JFA_SEED = helper("fx_jfa_seed", "_jfa_seed.wgsl", {"src": Input()}, {"seeds": Output(PortType.COLOR)},
                   {"threshold": ParamKind.FLOAT, "channel": ParamKind.INT}, libs=("_jfa_lib.wgsl",))
_JFA_FLOOD = helper("fx_jfa_flood", "_jfa_flood.wgsl", {"seeds": Input(PortType.COLOR)},
                    {"seeds": Output(PortType.COLOR)}, {"step": ParamKind.INT}, libs=("_jfa_lib.wgsl",))


def first_jump(w: int, h: int, reach: float | None = None) -> int:
    """The first step of a jump flooding at w x h: half the image, or the power of 2 past `reach` (pixels) when
    the seeds are needed only that far."""
    step = max(w, h) // 2
    return step if reach is None else min(step, 1 << (math.ceil(reach) + 2).bit_length())


def nearest_seeds(ctx, job, src, threshold=0.5, channel=0, reach=None):
    """A rgba32f texture at the job's size: per pixel, the nearest pixel above `threshold` (xy) and
    the nearest one below it (zw), -1 when there is none. `src` = (texture, is_gray); channel 0 reads
    red (luminance for color: see _jfa_seed.wgsl), 1 alpha. The passes run on the seeds packed in 16 bit integers
    (_jfa_lib.wgsl: exact coordinates, half the traffic), the last one unpacks them. `reach` (pixels): the
    caller needs the nearest seeds only that far, so the jumps start at the power of 2 past it (fewer passes);
    farther seeds may be missed (-1) or not the nearest."""
    w, h = job.size
    a, b = ctx.scratch(job, False, "16"), ctx.scratch(job, False, "16")
    ctx.dispatch(_JFA_SEED, job, {"threshold": float(threshold), "channel": channel}, srcs={"src": src},
                 dsts={"seeds": a})
    step = first_jump(w, h, reach)
    while step >= 1:
        ctx.dispatch(_JFA_FLOOD, job, {"step": step}, srcs={"seeds": (a, False)}, dsts={"seeds": b})
        a, b, step = b, a, step // 2
    # One extra pass at step 1 fixes most of jump flooding's rare misses.
    ctx.dispatch(_JFA_FLOOD, job, {"step": 1}, srcs={"seeds": (a, False)}, dsts={"seeds": b})
    out = ctx.scratch(job, False)
    ctx.dispatch(_JFA_FLOOD, job, {"step": 0}, srcs={"seeds": (b, False)}, dsts={"seeds": out})
    return out


# --- separable gaussian blur --------------------------------------------------------------------------

# Gaussian radius (output pixels at 256, ~2 sigma) of Blur HQ per unit of its Intensity, for the nodes that
# blur with it: sigma = 0.54 x Intensity.
BLUR_HQ_RADIUS = 1.078

_GAUSS = helper("fx_gauss", "_gauss.wgsl", {"src": Input()}, {"out": Output()},
                {"radius": ParamKind.FLOAT, "vertical": ParamKind.INT, "channel": ParamKind.INT,
                 "shift": ParamKind.FLOAT2})


def gauss(ctx, job, src, radius_px, channel=0, shift=(0.0, 0.0), gray=True, size=None):
    """Gaussian blur of `src` (texture, is_gray), radius (~2 sigma) in output pixels. channel: 0 the
    image as is, 1 its alpha, 2 its luminance (as grayscale). `shift` (UV) offsets the first read."""
    a = ctx.scratch(job, gray, size=size)
    b = ctx.scratch(job, gray, size=size)
    common = {"radius": float(radius_px)}
    ctx.dispatch(_GAUSS, job, {**common, "vertical": 0, "channel": channel, "shift": list(shift)},
                 srcs={"src": src}, dsts={"out": a}, size=size)
    ctx.dispatch(_GAUSS, job, {**common, "vertical": 1, "channel": 0, "shift": [0.0, 0.0]},
                 srcs={"src": (a, gray)}, dsts={"out": b}, size=size)
    return b


# --- inclusive prefix sums (Hillis-Steele, or of a higher radix), no wrap-around ---------------------------

_SCAN = helper("fx_scan", "_scan.wgsl", {"src": Input()}, {"out": Output()},
               {"step": ParamKind.INT, "vertical": ParamKind.INT, "radix": ParamKind.INT})


def prefix_sum(ctx, job, tex, gray, axes=(0, 1), size=None, radix=2):
    """Inclusive prefix sums of `tex` along x (0) and/or y (1), from the top-left pixel: passes adding `radix` - 1
    values each (a higher radix: fewer passes, the sums in another order)."""
    w, h = size or job.size
    bufs = [ctx.scratch(job, gray, size=size), ctx.scratch(job, gray, size=size)]
    cur, i = tex, 0
    for axis in axes:
        step, n = 1, (w if axis == 0 else h)
        while step < n:
            dst = bufs[i % 2]
            ctx.dispatch(_SCAN, job, {"step": step, "vertical": axis, "radix": radix}, srcs={"src": (cur, gray)},
                         dsts={"out": dst}, size=size)
            cur, i, step = dst, i + 1, step * radix
    return cur


# --- min / max over the whole image by 2x2 reduction -----------------------------------------------------

_REDUCE = helper("fx_reduce", "_reduce.wgsl", {"src": Input(PortType.COLOR)}, {"out": Output(PortType.COLOR)})


def reduce_max(ctx, job, tex):
    """Reduce a rgba32f texture to 1x1 by per-channel max (store minima negated to get them)."""
    w, h = tex.width, tex.height
    cur = tex
    while w > 1 or h > 1:
        w, h = max(1, (w + 1) // 2), max(1, (h + 1) // 2)
        out = ctx.scratch(job, False, size=(w, h))
        ctx.dispatch(_REDUCE, job, {}, srcs={"src": (cur, False)}, dsts={"out": out}, size=(w, h))
        cur = out
    return cur
