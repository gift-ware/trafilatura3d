"""Shape detection on an atlas opacity map, on the GPU: connected components by label propagation
with pointer jumping (each opaque texel carries the coordinates of the smallest texel of its
component), then a table of the components (bounding box, area, root) in raster order of their
roots, and a list of those not smaller than a threshold. Runs at a working resolution of at most
512 px (detection only; the shapes are then sampled from the full-resolution inputs)."""

import math

from trafilatura3d.nodes.base import Input, Output, ParamKind
from trafilatura3d.nodes.materials._base import COLOR, GRAY, Pass, passes_of, with_tiling

MAX_SHAPES = 256
MAX_RES = 512


# Shape Detection Precision: the label propagation runs x1, x2, x4 passes.
PRECISION = ("Simple or small shapes", "Complex or big shapes", "No failure mode")


def detection_passes(node, opacity_port: str):
    return passes_of(node, lambda n: (
        Pass(n, "lbl_init", "atlas_label_init.wgsl", {opacity_port: n.inputs[opacity_port]}, {"lbl": Output(COLOR)},
             includes=(), subst={"$OPACITY": opacity_port}),
        Pass(n, "lbl_step", "atlas_label_step.wgsl", {"lbl": Input(COLOR)}, {"out": Output(COLOR)}, includes=()),
        Pass(n, "table", "atlas_table.wgsl", {"lbl": Input(COLOR)}, {"table": Output(COLOR)}, includes=()),
        Pass(n, "list", "atlas_list.wgsl", {"table": Input(COLOR)}, {"list": Output(COLOR)}, includes=(),
             internal={"min_size": ParamKind.FLOAT})), "atlas")


def detect(node, ctx, job, opacity_port: str, res: int, iterations_scale: int, min_size: float) -> dict:
    """srcs for the consumer pass: `lbl` (labels), `table` (MAX_SHAPES x 2), `list` (MAX_SHAPES x 1)."""
    init, step, table, lst = detection_passes(node, opacity_port)
    res = max(8, min(MAX_RES, res))
    size = (res, res)
    flat = with_tiling(job, 0)  # shapes do not continue across the atlas edges
    a, b = ctx.scratch(job, False, size=size), ctx.scratch(job, False, size=size)
    ctx.dispatch(init, flat, job.values, dsts={"lbl": a}, size=size)
    for _ in range((2 * int(math.log2(res)) + 8) * iterations_scale):
        ctx.dispatch(step, flat, job.values, srcs={"lbl": (a, False)}, dsts={"out": b}, size=size)
        a, b = b, a
    t = ctx.scratch(job, False, size=(MAX_SHAPES, 2))
    ctx.dispatch(table, flat, job.values, srcs={"lbl": (a, False)}, dsts={"table": t}, size=(MAX_SHAPES, 1))
    lt = ctx.scratch(job, False, size=(MAX_SHAPES, 1))
    ctx.dispatch(lst, flat, {**job.values, "min_size": min_size}, srcs={"table": (t, False)}, dsts={"list": lt},
                 size=(MAX_SHAPES, 1))
    return {"lbl": (a, False), "table": (t, False), "list": (lt, False)}
