from trafilatura3d.nodes.materials._base import F, px256
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, BlurredMaskNode


class EdgeBlur(BlurredMaskNode):
    """Convex edges of the curvature, blurred by Blur Radius (pixels at 256 px), then levelled."""

    type = "edge_blur"
    label = "Edge Blur"
    catalog = "edge_blur"
    inputs = {"curvature": CURV, "mask": MASK}
    pre_inputs = ("curvature",)
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "blur_radius": F(2.0, "Blur Radius", 0.0, 8.0),
    }

    def blur_sigma(self, job):
        return job.values["blur_radius"] * px256(job)
