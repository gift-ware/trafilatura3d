from trafilatura3d.nodes.materials._base import F
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, MaskNode


class EdgeWear(MaskNode):
    """Wear on object edges from the curvature: Threshold sets how much of the convex range wears,
    Edges Width how full the worn band is, Disorder breaks its smoothness with noise."""

    type = "edge_wear"
    label = "Edge Wear"
    catalog = "edge_wear"
    inputs = {"curvature": CURV, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "threshold": F(1.0, "Threshold", 0.0, 1.0),
        "edges_width": F(0.1, "Edges Width", 0.0, 1.0),
        "disorder": F(0.0, "Disorder", 0.0, 1.0),
    }
