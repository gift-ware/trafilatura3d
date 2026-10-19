from trafilatura3d.nodes.materials._base import F
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, MaskNode


class EdgeDamages(MaskNode):
    """Damage on raised convex edges (curvature), spared where AO is occluded: chips at low Damages
    Intensity, chaotic scratches at high."""

    type = "edge_damages"
    label = "Edge Damages"
    catalog = "edge_damages"
    inputs = {"curvature": CURV, "ambient_occlusion": AO, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "damages_intensity": F(0.24, "Damages Intensity", 0.0, 1.0),
    }
