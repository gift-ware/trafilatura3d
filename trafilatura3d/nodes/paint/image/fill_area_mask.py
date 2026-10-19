from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, B, E, F, Filter, mesh_inputs

from ._fill_area import DETECTION, UV_BORDERS, areas
from ._passes import helper


class FillAreaMask(Filter):
    """Fills the areas closed by the mask's outlines with white. Outline (Area Detection): the value at or above
    Area Detection Threshold, or (Edges) a jump of at least the threshold between neighbours. An area is closed
    when it cannot reach the texture's border without crossing the outline (see _fill_area.py). UV Border
    Behavior Exclude Outside UVs never fills the pixels outside the UV islands (UV Islands coverage below UV Border
    Detection Threshold). Debug Area Detection shows the outline in white and the areas to fill in dark gray."""
    type = "paint_fill_area_mask"
    label = "Fill Area Mask"
    catalog = "painter.filter.fill_area_mask"
    doc = DOC + "filters/standard-filters/fill-area-mask"
    inputs = {"input": Input(label="Input"), **mesh_inputs("uv_islands")}
    params = {"area_detection": E("Area Detection", DETECTION),
              "area_detection_threshold": F("Area Detection Threshold", 0.5, 0.0, 1.0),
              "debug_area_detection": B("Debug Area Detection", False),
              "uv_border_behavior": E("UV Border Behavior", UV_BORDERS),
              "uv_border_detection_threshold": F("UV Border Detection Threshold", 0.5, 0.0, 1.0)}

    def encode(self, ctx, job):
        v = job.values
        scan, outline = areas(ctx, job, v["area_detection"], v["area_detection_threshold"], False)
        ctx.dispatch(FINAL, job, v, srcs={"scan": (scan, False), "outline": (outline, True)})


FINAL = helper("paint_fill_area_mask_final", "fill_area_mask_final.wgsl",
               {"input": Input(), "uv_islands": FillAreaMask.inputs["uv_islands"], "scan": Input(PortType.COLOR),
                "outline": Input(PortType.GRAY)}, {"output": Output()}, FillAreaMask.params, main=True)
