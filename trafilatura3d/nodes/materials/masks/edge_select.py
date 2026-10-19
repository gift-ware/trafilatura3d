from trafilatura3d.nodes.materials._base import F
from trafilatura3d.nodes.materials.masks._common import CURV, MASK, MaskNode


class EdgeSelect(MaskNode):
    """Selects edges from the curvature: convex and concave separately, each with a width (how far
    from the sharpest values it reaches), a softness and a maximum intensity; Level and Contrast
    then act on both."""

    type = "edge_select"
    label = "Edge Select"
    catalog = "edge_select"
    inputs = {"curvature": CURV, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "convex_edges_width": F(0.0, "Convex Edges Width", 0.0, 1.0, group="Convex"),
        "convex_softness": F(0.1, "Convex Softness", 0.0, 1.0, group="Convex"),
        "convex_intensity": F(1.0, "Convex Intensity", 0.0, 1.0, group="Convex"),
        "concave_edges_width": F(0.0, "Concave Edges Width", 0.0, 1.0, group="Concave"),
        "concave_softness": F(0.1, "Concave Softness", 0.0, 1.0, group="Concave"),
        "concave_intensity": F(1.0, "Concave Intensity", 0.0, 1.0, group="Concave"),
    }
