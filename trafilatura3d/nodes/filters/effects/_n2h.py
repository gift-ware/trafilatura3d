"""Normal map to height map on the GPU: the normals' slopes are integrated by solving the Poisson
equation (laplacian of h = divergence of the slopes) with Jacobi iterations on a pyramid, coarse to
fine: each level starts from the upsampled coarser solution, so low frequencies come from the small
levels and the iterations only fix the details. Shared by Normal to Height and Normal To Height HQ."""

from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType

from ._fx import helper, reduce_max

_SLOPES = helper("fx_n2h_slopes", "_n2h_slopes.wgsl", {"input": Input(PortType.COLOR)},
                 {"out": Output(PortType.COLOR)}, {"directx": ParamKind.INT})
_DOWN = helper("fx_down2", "_down.wgsl", {"src": Input()}, {"out": Output()})
_UP = helper("fx_up", "_up.wgsl", {"src": Input()}, {"out": Output()})
_JACOBI = helper("fx_n2h_jacobi", "_n2h_jacobi.wgsl", {"h": Input(PortType.GRAY), "g": Input(PortType.COLOR)},
                 {"out": Output(PortType.GRAY)})
_PREP = helper("fx_minmax_prep", "_minmax_prep.wgsl", {"src": Input()}, {"out": Output(PortType.COLOR)})


def integrate(ctx, job, directx: bool, iterations: int, coarsest: int, keep: tuple[int, ...]):
    """Returns (height, {level: its solution, smaller} for the levels in `keep`); heights in height
    units per UV width, at the job's size (level k = 1/2^k of it, clamped to the coarsest)."""
    w, h = job.size
    sizes = [(w, h)]
    while min(sizes[-1]) > coarsest:
        sizes.append((max(1, sizes[-1][0] // 2), max(1, sizes[-1][1] // 2)))
    g = [ctx.scratch(job, False)]
    ctx.dispatch(_SLOPES, job, {"directx": int(directx)}, dsts={"out": g[0]})
    for s in sizes[1:]:
        t = ctx.scratch(job, False, size=s)
        ctx.dispatch(_DOWN, job, {}, srcs={"src": (g[-1], False)}, dsts={"out": t}, size=s)
        g.append(t)
    cur = None
    kept = {}
    for level in range(len(sizes) - 1, -1, -1):
        s = sizes[level]
        a, b = ctx.scratch(job, True, size=s), ctx.scratch(job, True, size=s)
        if cur is None:
            ctx.dispatch(_UP, job, {}, srcs={"src": (ctx.ev.constant((0.0, 0.0, 0.0, 1.0)), True)},
                         dsts={"out": a}, size=s)
        else:
            ctx.dispatch(_UP, job, {}, srcs={"src": (cur, True)}, dsts={"out": a}, size=s)
        n = iterations * (2 if level == len(sizes) - 1 else 1)
        for _ in range(n):
            ctx.dispatch(_JACOBI, job, {}, srcs={"h": (a, True), "g": (g[level], False)}, dsts={"out": b}, size=s)
            a, b = b, a
        cur = a
        for k in keep:
            if level == min(k, len(sizes) - 1):
                kept[k] = cur
    return cur, kept


_COMBINE = helper("fx_n2h_combine", "_n2h_combine.wgsl",
                  {"h": Input(PortType.GRAY), "mid": Input(PortType.GRAY), "low": Input(PortType.GRAY)},
                  {"out": Output(PortType.GRAY)}, {"balance": ParamKind.FLOAT3})


def relief(ctx, job, directx: bool, iterations: int, coarsest: int, balance, stats: bool):
    """The integrated height with Relief Balance applied, three bands weighed by 2 x balance (low,
    mid, high; 0.5 = as integrated): the low band is the solution of the 1/16 level, the mid one what
    the 1/4 level adds, the high one the rest. As a gray scratch centred on 0.5 (one unit of slope per
    25.6 px at 256 = one unit of height, the inverse of Normal Sobel at intensity 1), plus min/max
    stats when asked."""
    hi, kept = integrate(ctx, job, directx, iterations, coarsest, (4, 2))
    out = ctx.scratch(job, True)
    ctx.dispatch(_COMBINE, job, {"balance": [float(b) for b in balance]},
                 srcs={"h": (hi, True), "mid": (kept[2], True), "low": (kept[4], True)}, dsts={"out": out})
    mm = None
    if stats:
        prep = ctx.scratch(job, False)
        ctx.dispatch(_PREP, job, {}, srcs={"src": (out, True)}, dsts={"out": prep})
        mm = reduce_max(ctx, job, prep)
    return out, mm
