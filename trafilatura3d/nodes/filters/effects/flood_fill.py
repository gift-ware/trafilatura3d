from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

from ._fx import Effect, helper, prefix_sum

TRADE_OFF = ("Simple or small shapes", "Complex or big shapes", "No failure mode")
ADVANCED = ("Display advanced Parameters and Output", "Hide advanced Parameters and Output")
F32 = {"output_format": {"mode": "absolute", "value": "32f"}}
RADIX = 8  # values each prefix sum pass adds per texel, + 1 (fewer passes)
PARENT = {"parent": "atomic<u32>"}
BOXES = {**PARENT, "box": "atomic<u32>"}
BOX = 6  # FF_BOX of _ff_cc.wgsl: u32 values per shape in `box`


class FloodFill(Effect):
    """Finds every connected white shape of a binary mask (cut at 0.5, 4-connected, across the
    tiling edges) on the GPU and encodes its bounding box for the other Flood Fill nodes.

    Connected components by union-find on a buffer (_ff_cc.wgsl): each white pixel points to a pixel of its shape
    with a smaller index; joining a pixel with its neighbours above and on the left hooks the larger root under the
    smaller one with an atomic min, so the root ends as the shape's first pixel (row-major) and the passes are a fixed
    few whatever the shapes' form. The roots then take a slot each and the shapes are counted (one value read back), so
    the boxes' buffer holds the shapes, not the pixels. The box comes from atomic min / max on the shape's values,
    taken where a run ends, in a frame whose seam lies in a sector of columns (rows) the shape does not cover: a shape
    across the tiling edge gets its own box. The result is exact, so the Safety/Speed trade-off, its Override and the Advanced options (which only show or
    hide parameters and the second output in the interface) leave it unchanged. Shapes must be separated by black, or
    they merge.

    Output (32F): R, G = the pixel's position inside its shape's bounding box (0-1), B, A = the box's
    width and height in UV; black outside shapes. Special Shape Info (the advanced output): R, G
    = the UV of the shape's first pixel (row-major), B = the shape's index (0 at the top left,
    row-major by first pixel), A = the number of shapes."""

    type = "flood_fill"
    label = "Flood Fill"
    category = "Filters/Effects"
    catalog = "flood_fill"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.COLOR, "Output"),
               "special_shape_info": Output(PortType.COLOR, "Special Shape Info")}
    base_defaults = F32
    params = {
        "safety_speed_trade_off": Param(ParamKind.ENUM, TRADE_OFF[1], "Safety/Speed trade-off", choices=TRADE_OFF),
        "advanced_options": Param(ParamKind.ENUM, ADVANCED[1], "Advanced options", choices=ADVANCED),
        "override_safety_speed_trade_off": Param(ParamKind.INT, -1, "Override Safety/Speed trade-off", -1, 100),
    }

    def encode(self, ctx, job):
        w, h = job.size
        parent, count = ctx.buffer(4 * w * h), ctx.buffer(4)
        for node in (INIT, MERGE, COMPRESS):
            ctx.dispatch(node, job, {}, bufs={"parent": parent})
        ctx.dispatch(SLOTS, job, {}, bufs={"parent": parent, "count": count})
        # The shapes counted once the GPU is there (this encode is a generator: the event loop runs meanwhile in a
        # cook in steps), then their values.
        yield ctx.flush()
        bufs = {"parent": parent, "box": ctx.buffer(4 * BOX * int(ctx.read(count)[0]))}
        for node in (BOX_INIT, SECTORS):
            ctx.dispatch(node, job, {}, bufs=bufs)
        roots = ctx.scratch(job, True)
        ctx.dispatch(EXTENT, job, {}, dsts={"out": roots}, bufs=bufs)
        # Shape indices: prefix count of the shapes' first pixels, by row then over the row totals.
        rows = prefix_sum(ctx, job, roots, True, axes=(0,), radix=RADIX)  # counts: exact in any order
        totals = ctx.scratch(job, True, size=(1, h))
        ctx.dispatch(ROW_TOTALS, job, {}, srcs={"rows": (rows, True)}, dsts={"out": totals}, size=(1, h))
        column = prefix_sum(ctx, job, totals, True, axes=(1,), size=(1, h), radix=RADIX)
        ctx.dispatch(FINAL, job, job.values, srcs={"rows": (rows, True), "column": (column, True)}, bufs=bufs)


def _pass(type_: str, shader: str, buffers: dict, inputs: dict | None = None, outputs: dict | None = None) -> Node:
    return helper(type_, shader, inputs or {}, outputs or {}, libs=("_ff_cc.wgsl",), buffers=buffers)


INIT = _pass("fx_ff_init", "_ff_init.wgsl", PARENT, {"input": Input(PortType.GRAY)})
MERGE = _pass("fx_ff_merge", "_ff_merge.wgsl", PARENT)
COMPRESS = _pass("fx_ff_compress", "_ff_compress.wgsl", PARENT)
SLOTS = _pass("fx_ff_slots", "_ff_slots.wgsl", {**PARENT, "count": "atomic<u32>"})
BOX_INIT = _pass("fx_ff_box_init", "_ff_box_init.wgsl", BOXES)
SECTORS = _pass("fx_ff_sectors", "_ff_sectors.wgsl", BOXES)
EXTENT = _pass("fx_ff_extent", "_ff_extent.wgsl", BOXES, outputs={"out": Output(PortType.GRAY)})
ROW_TOTALS = helper("fx_ff_row_totals", "_ff_row_totals.wgsl", {"rows": Input(PortType.GRAY)},
                    {"out": Output(PortType.GRAY)})


class _Final(Effect):
    type = "fx_flood_fill_final"
    registered = False
    libs = ("_ff_cc.wgsl",)
    buffers = BOXES
    inputs = {"rows": Input(PortType.GRAY), "column": Input(PortType.GRAY)}
    outputs = FloodFill.outputs
    params = FloodFill.params


FINAL = _Final()
