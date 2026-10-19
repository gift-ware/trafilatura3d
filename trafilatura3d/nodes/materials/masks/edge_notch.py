from trafilatura3d.nodes.materials._base import F, px256
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, BlurredMaskNode


class EdgeNotch(BlurredMaskNode):
    """Convex edges interrupted by small notches cut by a fine noise; Radius blurs the
    edges first (a Blur HQ of the convex curvature)."""

    type = "edge_notch"
    label = "Edge Notch"
    catalog = "edge_notch"
    inputs = {"curvature": CURV, "mask": MASK}
    pre_inputs = ("curvature",)
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "radius": F(0.0, "Radius", 0.0, 8.0),
    }

    def blur_sigma(self, job):
        return job.values["radius"] * px256(job)
