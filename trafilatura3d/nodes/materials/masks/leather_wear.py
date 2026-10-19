from trafilatura3d.nodes.materials._base import B, F, gray_in
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, MaskNode


class LeatherWear(MaskNode):
    """Wear with a leather pattern: convex edges (weighted by Curvature Weight) wear first, the
    built-in leather grain (wrinkle network of cells) or the Grunge Input spreads it between edges,
    occluded areas (AO) are spared."""

    type = "leather_wear"
    label = "Leather Wear"
    catalog = "leather_wear"
    inputs = {"curvature": CURV, "ambient_occlusion": AO, "grunge_input": gray_in("Grunge Input", 0.5),
              "mask": MASK}
    params = {
        "wear_level": F(0.75, "Wear Level", 0.0, 1.0),
        "wear_contrast": F(0.5, "Wear Contrast", 0.0, 1.0),
        "grunge_amount": F(0.25, "Grunge Amount", 0.0, 1.0),
        "ambient_occlusion_masking": F(0.5, "Ambient Occlusion Masking", 0.0, 1.0),
        "curvature_weight": F(0.25, "Curvature Weight", 0.0, 1.0),
        "use_custom_grunge": B(False, "Use Custom Grunge"),
    }
