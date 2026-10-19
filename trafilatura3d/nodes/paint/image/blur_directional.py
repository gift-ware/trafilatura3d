from trafilatura3d.nodes.atomic.directional_blur import directional_blur
from trafilatura3d.nodes.filters.basic._lib import source
from trafilatura3d.nodes.paint._lib import DOC, F, Filter


class BlurDirectional(Filter):
    """Blur stretched along Angle (turns, clockwise from +X): Blur Intensity (pixels at 256) on each side along
    the angle, Blur Intensity x (1 - Intensity) across it (Intensity 1: a pure motion blur, 0: the same blur in
    every direction). Two line blurs (the Directional Blur node's), along then across."""
    type = "paint_blur_directional"
    label = "Blur Directional"
    catalog = "painter.filter.blur_directional"
    doc = DOC + "filters/standard-filters/blur-directional"
    params = {"blur_intensity": F("Blur Intensity", 8.0, 0.0, 64.0), "angle": F("Angle", 0.0, 0.0, 1.0),
              "intensity": F("Intensity", 1.0, 0.0, 1.0)}

    def encode(self, ctx, job):
        v = job.values
        gray = job.is_gray("output")
        along = directional_blur(ctx, job, source(ctx, job, "input"), gray, v["blur_intensity"], v["angle"],
                                 ctx.scratch(job, gray))
        across = v["blur_intensity"] * (1.0 - min(max(v["intensity"], 0.0), 1.0))
        directional_blur(ctx, job, (along, gray), gray, across, v["angle"] + 0.25)
