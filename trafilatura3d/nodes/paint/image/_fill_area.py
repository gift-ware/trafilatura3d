"""Area detection of Fill Area Mask / Color: which pixels are outline, which are outside every closed outline.

The outline is the input above the threshold (or its edges); the exterior is every non-outline pixel connected to
the texture's border without crossing the outline (UV space: the border is outside the islands, nothing wraps).
It spreads by rounds of segmented OR-scans (Hillis-Steele, log2(size) doubling passes) along the rows, then the
columns: a scan reaches a whole run of open pixels at once, so a round follows any shape that needs one turn,
and ROUNDS rounds follow outlines that wind up to ROUNDS times. What stays is outline or inside."""

from __future__ import annotations

from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.filters.effects._fx import source

from ._passes import helper

ROUNDS = 8
DETECTION = ("Threshold", "Edges")
UV_BORDERS = ("Ignore", "Exclude Outside UVs")

_SCAN = {"scan": Input(PortType.COLOR)}
INIT = helper("paint_fill_area_init", "fill_area_init.wgsl", {"input": Input()},
              {"scan": Output(PortType.COLOR), "outline": Output(PortType.GRAY)},
              internal={"thr_": ParamKind.FLOAT, "edges_": ParamKind.INT, "alpha_": ParamKind.INT})
SCAN = helper("paint_fill_area_scan", "fill_area_scan.wgsl", _SCAN, {"scan": Output(PortType.COLOR)},
              internal={"step_": ParamKind.INT, "vertical_": ParamKind.INT})
MERGE = helper("paint_fill_area_merge", "fill_area_merge.wgsl", {**_SCAN, "outline": Input(PortType.GRAY)},
               {"scan": Output(PortType.COLOR)})


def areas(ctx, job, detection: str, threshold: float, use_alpha: bool):
    """(scan, outline): scratch textures, the exterior flag in scan's red, outline = 1 on the outline. `use_alpha`
    detects on the input's alpha (Fill Area Color) instead of its value; a grayscale input has no alpha."""
    w, h = job.size
    alpha = use_alpha and not source(ctx, job, "input")[1]
    bufs = [ctx.scratch(job, False), ctx.scratch(job, False)]
    outline = ctx.scratch(job, True)
    ctx.dispatch(INIT, job, {"thr_": float(threshold), "edges_": DETECTION.index(detection), "alpha_": int(alpha)},
                 dsts={"scan": bufs[0], "outline": outline})
    cur = 0

    def flip(node, values, extra=None):
        nonlocal cur
        ctx.dispatch(node, job, values, srcs={"scan": (bufs[cur], False), **(extra or {})},
                     dsts={"scan": bufs[1 - cur]})
        cur = 1 - cur

    for _ in range(ROUNDS):
        for vertical, n in ((0, w), (1, h)):
            step = 1
            while step < n:
                flip(SCAN, {"step_": step, "vertical_": vertical})
                step *= 2
            flip(MERGE, {}, {"outline": (outline, True)})
    return bufs[cur], outline
