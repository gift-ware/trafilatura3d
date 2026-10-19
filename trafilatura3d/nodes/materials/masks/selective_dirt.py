from trafilatura3d.nodes.materials._base import B, F
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, VARIATION, MaskNode


class SelectiveDirt(MaskNode):
    """A simple dirt on convex edges (curvature), broken up by a variation (built-in grunge or the
    Variation Mask)."""

    type = "selective_dirt"
    label = "Selective Dirt"
    catalog = "selective_dirt"
    inputs = {"curvature": CURV, "variation_mask": VARIATION, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "variation": F(0.0, "Variation", 0.0, 1.0),
        "override_variation_mask": B(False, "Override variation mask"),
    }
