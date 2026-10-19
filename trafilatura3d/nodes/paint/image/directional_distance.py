from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.filters.effects._fx import source
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, F, Filter

from ._passes import helper


class DirectionalDistance(Filter):
    """A distance gradient travelling from the input's bright areas towards Angle (turns, 0 = right (1, 0),
    clockwise: 0.25 = down): each pixel takes max over t of value(p - t x direction) - t / Distance, i.e. a shape
    of value 1 leaves a linear trail falling to 0 at Distance (normalized: 1 = the image's shorter side). The
    Distance Map scales Distance per pixel by Distance Map Multiplier (no effect unconnected); Contrast sharpens
    the falloff (result^(1 + 7 x Contrast)). Computed in log2(Distance in pixels) passes, each looking twice as
    far back as the previous one."""
    type = "paint_directional_distance"
    label = "Directional Distance"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.directional_distance"
    doc = DOC + "filters/advanced-filters/directional-distance"
    inputs = {"input": Input(label="Input"),
              "distance_map": Input(PortType.GRAY, "Distance Map", (1.0, 1.0, 1.0, 1.0))}
    params = {"distance": F("Distance", 0.15, 0.0, 1.0), "angle": F("Angle", 0.25, 0.0, 1.0),
              "contrast": F("Contrast", 0.0, 0.0, 1.0),
              "distance_map_multiplier": F("Distance Map Multiplier", 1.0, 0.0, 1.0)}

    def encode(self, ctx, job):
        v = job.values
        reach = v["distance"] * min(job.size)
        state = source(ctx, job, "input")
        bufs = [ctx.scratch(job, True), ctx.scratch(job, True)]
        step, i = 1.0, 0
        while step <= reach:
            ctx.dispatch(STEP, job, {**v, "step_": step}, srcs={"state": state}, dsts={"out": bufs[i % 2]})
            state, step, i = (bufs[i % 2], True), step * 2.0, i + 1
        ctx.dispatch(FINAL, job, v, srcs={"state": state})


STEP = helper("paint_directional_distance_step", "directional_distance_step.wgsl",
              {"state": Input(), "distance_map": Input(PortType.GRAY, default=(1.0, 1.0, 1.0, 1.0))},
              {"out": Output(PortType.GRAY)}, DirectionalDistance.params, {"step_": ParamKind.FLOAT})
FINAL = helper("paint_directional_distance_final", "directional_distance_final.wgsl",
               {"input": Input(), "state": Input()}, {"output": Output()}, DirectionalDistance.params, main=True)
