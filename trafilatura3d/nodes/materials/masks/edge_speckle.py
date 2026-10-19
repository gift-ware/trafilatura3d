from trafilatura3d.nodes.materials._base import B, F
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, VARIATION, MaskNode


class EdgeSpeckle(MaskNode):
    """Edges dotted with speckles: Edge selection moves the effect from all edges to convex ones,
    Variation masks it with a large-scale breakup (built-in or the Variation Mask)."""

    type = "edge_speckle"
    label = "Edge Speckle"
    catalog = "edge_speckle"
    inputs = {"curvature": CURV, "variation_mask": VARIATION, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "edge_selection": F(0.5, "Edge selection", 0.0, 1.0),
        "variation": F(0.0, "Variation", 0.0, 1.0),
        "override_variation_mask": B(False, "Override variation mask"),
    }
