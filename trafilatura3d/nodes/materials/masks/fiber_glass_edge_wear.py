from trafilatura3d.nodes.materials._base import B, F, gray_in
from trafilatura3d.nodes.materials.masks._common import AO, CURV, MASK, POS, WSN, EdgeWearNode


class FiberGlassEdgeWear(EdgeWearNode):
    """Fibreglass wear: convex edges (Curvature Weight) smoothed by Edges Smoothness, a woven fibre pattern
    (built-in, or the Grunge Input; optionally triplanar from Position and World Space Normal) revealing wear
    between edges; occluded areas (AO, by AO Masking) take wear away before Wear Level."""

    type = "fiber_glass_edge_wear"
    label = "Fiber Glass Edge Wear"
    catalog = "fiber_glass_edge_wear"
    inputs = {"curvature": CURV, "ambient_occlusion": AO, "grunge_input": gray_in("Grunge Input", 0.5),
              "mask": MASK, "world_space_normal": WSN, "position": POS}
    params = {
        "wear_level": F(0.75, "Wear Level", 0.0, 1.0),
        "wear_contrast": F(0.5, "Wear Contrast", 0.0, 1.0),
        "edges_smoothness": F(1.0, "Edges Smoothness", 0.0, 16.0),
        "grunge_amount": F(0.25, "Grunge Amount", 0.0, 1.0),
        "ambient_occlusion_masking": F(0.5, "Ambient Occlusion Masking", 0.0, 1.0),
        "curvature_weight": F(0.25, "Curvature Weight", 0.0, 1.0),
        "use_custom_grunge": B(False, "Use Custom Grunge"),
        "use_triplanar": B(False, "Use Triplanar"),
        "triplanar_blending_contrast": F(0.5, "Triplanar Blending Contrast", 0.0, 1.0),
    }
