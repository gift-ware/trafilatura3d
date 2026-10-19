from trafilatura3d.nodes.paint._lib import DOC, B, F, Filter, blur


class Blur(Filter):
    """Gaussian blur; Blur Wrap off clamps at the texture borders instead of wrapping around."""
    type = "paint_blur"
    label = "Blur"
    catalog = "painter.filter.blur"
    doc = DOC + "filters/standard-filters/blur"
    params = {"blur_intensity": F("Blur Intensity", 2.0, 0.0, 16.0), "blur_wrap": B("Blur Wrap", True)}

    def encode(self, ctx, job):
        blur(ctx, job, job.values["blur_intensity"], job.values["blur_wrap"])
