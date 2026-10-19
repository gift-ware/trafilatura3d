from trafilatura3d.nodes.materials._base import B, F, I, gray_in
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, POS, WSN, EdgeWearNode


class MetalEdgeWear(EdgeWearNode):
    """Metal edge wear: convex edges (Curvature Weight) smoothed by Edges Smoothness, chipped and scratched by a
    grunge (built-in at Grunge Scale, or the Grunge Input; optionally triplanar); occluded areas (AO, by AO Masking)
    scale the edges down before Wear Level."""

    type = "metal_edge_wear"
    label = "Metal Edge Wear"
    catalog = "metal_edge_wear"
    inputs = {"curvature": CURV, "ambient_occlusion": AO, "grunge_input": gray_in("Grunge Input", 0.5),
              "mask": MASK, "world_space_normal": WSN, "position": POS}
    params = {
        "wear_level": F(0.75, "Wear Level", 0.0, 1.0),
        "wear_contrast": F(0.5, "Wear Contrast", 0.0, 1.0),
        "edges_smoothness": F(1.0, "Edges Smoothness", 0.0, 16.0),
        "grunge_amount": F(0.4, "Grunge Amount", 0.0, 1.0),
        "grunge_scale": I(1, "Grunge Scale", 1, 16),
        "ambient_occlusion_masking": F(0.5, "Ambient Occlusion Masking", 0.0, 1.0),
        "curvature_weight": F(1.0, "Curvature Weight", 0.0, 1.0),
        "use_custom_grunge": B(False, "Use Custom Grunge"),
        "use_triplanar": B(False, "Use Triplanar"),
        "triplanar_blending_contrast": F(0.5, "Triplanar Blending Contrast", 0.0, 1.0),
    }
