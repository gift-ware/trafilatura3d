from trafilatura3d.nodes.materials._base import B, F
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, VARIATION, MaskNode


class EdgeDirt(MaskNode):
    """Dirt gathered around edges, mostly in concave creases and less on convex rims, from the
    curvature alone; Variation blends in a large-scale breakup (built-in, or the Variation Mask when
    overridden)."""

    type = "edge_dirt"
    label = "Edge Dirt"
    catalog = "edge_dirt"
    inputs = {"curvature": CURV, "variation_mask": VARIATION, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "variation": F(0.0, "Variation", 0.0, 1.0),
        "override_variation_mask": B(False, "Override variation mask"),
    }
