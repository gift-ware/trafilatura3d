from trafilatura3d.nodes.materials._base import B, F, px256
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, VARIATION, BlurredMaskNode


class PaintWear(BlurredMaskNode):
    """Paint chipping away at edges: convex edges spread by Radius, cut into chips by a cellular
    variation (built-in, or the Variation Mask), prevented in occluded areas (AO) by Occlusion."""

    type = "paint_wear"
    main_input = "curvature"  # the library graph's primary input: Relative to input follows it
    label = "Paint Wear"
    catalog = "paint_wear"
    inputs = {"ambient_occlusion": AO, "curvature": CURV, "variation_mask": VARIATION, "mask": MASK}
    pre_inputs = ("curvature",)
    final_also = ("curvature",)
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "occlusion": F(0.5, "Occlusion", 0.0, 1.0),
        "radius": F(1.0, "Radius", 0.0, 2.0),
        "variation": F(0.5, "Variation", 0.0, 1.0),
        "override_variation_mask": B(False, "Override variation mask"),
    }

    def blur_sigma(self, job):
        return job.values["radius"] * 6.0 * px256(job)
