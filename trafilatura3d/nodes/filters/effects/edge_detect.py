import math

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect, helper, nearest_seeds, scale


class EdgeDetect(Effect):
    """White lines along the contrast edges of an image: pixels that stand out of the mean of their 4 neighbours
    (smooth ramps are not edges) by the threshold Tolerance sets (0 = the slightest contrast), grown by
    Edge Width - 1 pixels at 256, then rounded by a closing of Edge Roundness pixels at 256 (grown, then
    shrunk back: inner corners fill in). Distances exact, by jump flooding. Invert swaps black and
    white."""

    type = "edge_detect"
    label = "Edge Detect"
    category = "Filters/Effects"
    catalog = "edge_detect"
    libs = ("_shape_sd.wgsl",)
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "edge_width": Param(ParamKind.FLOAT, 2.0, "Edge Width", 1.0, 16.0),
        "edge_roundness": Param(ParamKind.FLOAT, 4.0, "Edge Roundness", 0.0, 16.0),
        "invert": Param(ParamKind.BOOL, False, "Invert"),
        "tolerance": Param(ParamKind.FLOAT, 0.0, "Tolerance", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        v = job.values
        # The passes look only as far as they grow and shrink (_edge_dilate.wgsl, edge_detect.wgsl), a pixel more.
        close = math.ceil(max(v["edge_roundness"], 0.0)) * scale(job)
        grow = max(v["edge_width"] - 1.0, 0.0) * scale(job) + close
        edges = ctx.scratch(job, True)
        ctx.dispatch(CONTRAST, job, v, dsts={"out": edges})
        grown = ctx.scratch(job, True)
        ctx.dispatch(DILATE, job, v, srcs={"seeds": (nearest_seeds(ctx, job, (edges, True), reach=grow + 1), False)},
                     dsts={"out": grown})
        ctx.dispatch(FINAL, job, v, srcs={"seeds": (nearest_seeds(ctx, job, (grown, True), reach=close + 1), False)})


CONTRAST = helper("fx_edge_contrast", "_edge_contrast.wgsl", {"input": Input(PortType.GRAY)},
                  {"out": Output(PortType.GRAY)}, params=EdgeDetect.params)
DILATE = helper("fx_edge_dilate", "_edge_dilate.wgsl", {"seeds": Input(PortType.COLOR)},
                {"out": Output(PortType.GRAY)}, params=EdgeDetect.params, libs=("_shape_sd.wgsl",))


class _Final(Effect):
    type = "fx_edge_detect_final"
    registered = False
    libs = ("_shape_sd.wgsl",)
    inputs = {"seeds": Input(PortType.COLOR)}
    outputs = EdgeDetect.outputs
    params = EdgeDetect.params


FINAL = _Final()
