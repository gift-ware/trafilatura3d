import math

from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.paint._lib import DOC, E, F, helper, mesh_inputs, soft_blur
from trafilatura3d.nodes.paint.generators._common import GenNode, custom, extended, island_borders, mask_sides

STITCH = helper("gen_stitch_passes", "generators/stitch_passes.wgsl",
                {"a_": Input(PortType.COLOR), "b_": Input(PortType.COLOR)}, {"output": Output(PortType.COLOR)},
                internal={"pass_": ParamKind.INT}, main=True)


class AutoStitcher(GenNode):
    """Stitches along paths at Path Position (px at 256) from the borders of a region: UV Mask, the UV island borders
    (computed `uv_islands` map), inside the islands; Curvature, the hard edges (curvature >= 0.6), on both sides;
    Custom Input, the borders between its black and white areas, on both sides (a path never crosses them). Path
    Smoothness blurs the distance field before the path is taken (rounder paths). The path's arc length is computed
    on the GPU (successor links along the path's tangent, pointer jumping; loops are cut at one point), every pixel
    takes its nearest path pixel's length, and stitches are drawn every Stitch Size period: a rounded rectangle
    (Stitch Length along, Stitch Width across, Stitch Roundness from square to round), each rotated by a random angle
    up to Jitter. Output: a dome per stitch (1 at its centre)."""
    type = "paint_auto_stitcher"
    label = "Auto Stitcher"
    catalog = "painter.generator.auto_stitcher"
    doc = DOC + "generators/auto-stitcher"
    includes = ("generators/common",)
    inputs = {**mesh_inputs("curvature", "uv_islands"), "custom_input": custom("Custom Input", gray=True)}
    params = {"mask_mode": E("Mask Mode", ("UV Mask", "Curvature", "Custom Input")),
              "path_smoothness": F("Path Smoothness", 0.2, 0.0, 1.0),
              "path_position": F("Path Position", 4.0, 0.0, 32.0),
              "stitch_size": F("Stitch Size", 1.0, 0.1, 4.0), "stitch_width": F("Stitch Width", 0.25, 0.0, 1.0),
              "stitch_length": F("Stitch Length", 0.6, 0.0, 1.0),
              "stitch_roundness": F("Stitch Roundness", 0.5, 0.0, 1.0), "jitter": F("Jitter", 0.0, 0.0, 1.0)}
    STAGES = {"seeds_": Input(PortType.COLOR), "dist_": Input(PortType.GRAY), "path_": Input(PortType.COLOR),
              "rank_": Input(PortType.COLOR), "near_": Input(PortType.COLOR)}

    def encode(self, ctx, job):
        st, v = extended(self, "stages", self.STAGES, {"stage_": ParamKind.INT}), job.values
        w, h = job.size
        if v["mask_mode"] == "UV Mask":
            seeds = island_borders(ctx, job)
        else:
            region = ctx.scratch(job, True)
            ctx.dispatch(st, job, {**v, "stage_": 0}, dsts={"output": region})
            seeds = mask_sides(ctx, job, (region, True))
        dist = ctx.scratch(job, True)
        ctx.dispatch(st, job, {**v, "stage_": 1}, srcs={"seeds_": (seeds, False)}, dsts={"output": dist})
        srcs = {"dist_": (soft_blur(ctx, job, dist, True, v["path_smoothness"] * 8.0), True)}
        path = ctx.scratch(job, False)
        ctx.dispatch(st, job, {**v, "stage_": 2}, srcs=srcs, dsts={"output": path})
        # Arc length along the paths.
        jumps = math.ceil(math.log2(max(w * h, 2))) + 1
        succ = ctx.scratch(job, False)
        ctx.dispatch(STITCH, job, {"pass_": 0}, srcs={"a_": (path, False)}, dsts={"output": succ})
        a, b = succ, ctx.scratch(job, False)
        for _ in range(jumps):
            ctx.dispatch(STITCH, job, {"pass_": 1}, srcs={"a_": (a, False)}, dsts={"output": b})
            a, b = b, (ctx.scratch(job, False) if a is succ else a)
        links = ctx.scratch(job, False)
        ctx.dispatch(STITCH, job, {"pass_": 2}, srcs={"a_": (succ, False), "b_": (a, False)}, dsts={"output": links})
        a, b = links, ctx.scratch(job, False)
        for _ in range(jumps):
            ctx.dispatch(STITCH, job, {"pass_": 3}, srcs={"a_": (a, False)}, dsts={"output": b})
            a, b = b, a
        srcs |= {"path_": (path, False), "rank_": (a, False), "near_": (mask_sides(ctx, job, (path, False)), False)}
        ctx.dispatch(st, job, {**v, "stage_": 3}, srcs=srcs)
