import math

from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType
from trafilatura3d.nodes.filters.effects._fx import nearest_seeds
from trafilatura3d.nodes.paint._lib import DOC, B, C, E, F, I, Filter, mesh_inputs

from ._fill_area import DETECTION, UV_BORDERS, areas
from ._passes import helper, wrapping

COLOR_MODES = ("Fill Color", "Blur", "Diffusion")


class FillAreaColor(Filter):
    """Fills the areas closed by painted strokes (the outline: alpha at or above Area Detection Threshold, or with
    Edges a jump of alpha of at least the threshold; a grayscale input has no alpha: its value) with an opaque
    color: Fill Color; Blur, the color of the nearest stroke pixel (jump flooding) averaged over Blur Samples
    points of a disc of Blur Intensity (pixels at 256) around the pixel; Diffusion, the stroke colors
    spread inwards over Diffusion Iterations passes from coarse to fine steps. An area is closed when it cannot
    reach the texture's border without crossing a stroke (see _fill_area.py). UV Border Behavior Exclude Outside
    UVs never fills the pixels outside the UV islands (UV Islands coverage below UV Border Threshold). Debug Area
    Detection shows the outline in white and the areas to fill in dark gray."""
    type = "paint_fill_area_color"
    label = "Fill Area Color"
    catalog = "painter.filter.fill_area_color"
    doc = DOC + "filters/standard-filters/fill-area-color"
    inputs = {"input": Input(label="Input"), **mesh_inputs("uv_islands")}
    params = {"area_detection": E("Area Detection", DETECTION),
              "area_detection_threshold": F("Area Detection Threshold", 0.5, 0.0, 1.0),
              "debug_area_detection": B("Debug Area Detection", False),
              "uv_border_behavior": E("UV Border Behavior", UV_BORDERS),
              "uv_border_threshold": F("UV Border Threshold", 0.5, 0.0, 1.0),
              "color_mode": E("Color Mode", COLOR_MODES), "fill_color": C("Fill Color", (1.0, 1.0, 1.0)),
              "blur_intensity": F("Blur Intensity", 8.0, 0.0, 64.0), "blur_samples": I("Blur Samples", 32, 4, 128),
              "diffusion_iterations": I("Diffusion Iterations", 16, 1, 64)}

    def encode(self, ctx, job):
        v = job.values
        scan, outline = areas(ctx, job, v["area_detection"], v["area_detection_threshold"], True)
        known = {"scan": (scan, False), "outline": (outline, True)}
        colors = seeds = scan  # read only in Diffusion / Blur mode
        if v["color_mode"] == "Blur":
            seeds = nearest_seeds(ctx, wrapping(job, False), (outline, True))
        if v["color_mode"] == "Diffusion":
            bufs = [ctx.scratch(job, False), ctx.scratch(job, False)]
            ctx.dispatch(DIFFUSE, job, {"step_": 0}, srcs=known, dsts={"colors": bufs[0]})
            n = max(int(v["diffusion_iterations"]), 1)
            top = max(math.log2(max(job.size) / 8.0), 0.0)
            for k in range(n):
                step = max(1, round(2.0 ** (top * (1.0 - k / max(n - 1, 1)))))
                ctx.dispatch(DIFFUSE, job, {"step_": step}, srcs={**known, "colors": (bufs[k % 2], False)},
                             dsts={"colors": bufs[1 - k % 2]})
            colors = bufs[n % 2]
        ctx.dispatch(FINAL, job, v, srcs={**known, "colors": (colors, False), "seeds": (seeds, False)})


DIFFUSE = helper("paint_fill_area_diffuse", "fill_area_diffuse.wgsl",
                 {"input": Input(), "scan": Input(PortType.COLOR), "outline": Input(PortType.GRAY),
                  "colors": Input(PortType.COLOR)}, {"colors": Output(PortType.COLOR)}, internal={"step_": ParamKind.INT})
FINAL = helper("paint_fill_area_color_final", "fill_area_color_final.wgsl",
               {"input": Input(), "uv_islands": FillAreaColor.inputs["uv_islands"], "scan": Input(PortType.COLOR),
                "outline": Input(PortType.GRAY), "colors": Input(PortType.COLOR), "seeds": Input(PortType.COLOR)},
               {"output": Output()}, FillAreaColor.params, main=True)
