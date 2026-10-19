from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect, helper, prefix_sum
from .flood_fill import F32, RADIX, ROW_TOTALS


class FloodFillToIndex(Effect):
    """Numbers the Flood Fill shapes from 0 at the top left (row-major by each shape's first pixel):
    Normalized = index / number of shapes, Integer = the index itself (HDR, 32F by default).
    Shapes whose bounding box diagonal (UV) is smaller than Ignore Shape Smaller
    Than (and than a pixel) are left black and not counted; the indices are recounted on the GPU
    (prefix sums of the kept shapes' first pixels). Shape Count: the number of shapes kept,
    everywhere. Needs Flood Fill's Special Shape Info output."""

    type = "flood_fill_to_index"
    label = "Flood Fill to Index"
    category = "Filters/Effects"
    catalog = "flood_fill_to_index"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill_bbox": Input(PortType.COLOR, "Flood Fill Bbox"),
              "special_shape_info": Input(PortType.COLOR, "Special Shape Info")}
    outputs = {"shape_index": Output(PortType.GRAY, "Shape Index"),
               "shape_count": Output(PortType.GRAY, "Shape Count")}
    base_defaults = F32
    params = {
        "output": Param(ParamKind.ENUM, "Normalized", "Output", choices=("Normalized", "Integer")),
        "ignore_shape_smaller_than": Param(ParamKind.FLOAT, 0.0, "Ignore Shape Smaller Than", 0.0, 1.0),
    }
    unsupported = {"Show Flood Fill Data Table": "the table's layout is not documented, and writing one texel per "
                                                 "shape needs a scatter that texture-only passes cannot express"}

    def encode(self, ctx, job):
        src = job.inputs.get("special_shape_info")
        n = src[0].size if src else job.size
        keep = ctx.scratch(job, True, size=n)
        ctx.dispatch(KEEP, job, job.values, dsts={"out": keep}, size=n)
        rows = prefix_sum(ctx, job, keep, True, axes=(0,), size=n, radix=RADIX)  # counts: exact in any order
        totals = ctx.scratch(job, True, size=(1, n[1]))
        ctx.dispatch(ROW_TOTALS, job, {}, srcs={"rows": (rows, True)}, dsts={"out": totals}, size=(1, n[1]))
        column = prefix_sum(ctx, job, totals, True, axes=(1,), size=(1, n[1]), radix=RADIX)
        ctx.dispatch(FINAL, job, job.values, srcs={"rows": (rows, True), "column": (column, True)})


KEEP = helper("fx_ffi_keep", "_ffi_keep.wgsl", FloodFillToIndex.inputs, {"out": Output(PortType.GRAY)},
              params=FloodFillToIndex.params, libs=("_ff_read.wgsl",))


class _Final(Effect):
    type = "fx_flood_fill_to_index_final"
    registered = False
    libs = ("_ff_read.wgsl",)
    inputs = {**FloodFillToIndex.inputs, "rows": Input(PortType.GRAY), "column": Input(PortType.GRAY)}
    outputs = FloodFillToIndex.outputs
    params = FloodFillToIndex.params


FINAL = _Final()
