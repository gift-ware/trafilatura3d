from trafilatura3d.nodes.paint._lib import DOC, SEED, B, F, Generator, mesh_inputs

MICRO = {"Use Custom Grunge": "custom grunge maps and anchor points are not offered yet",
         "Micro Height": "micro details (custom maps) are not offered yet",
         "Micro Normal": "micro details (custom maps) are not offered yet",
         "Curvature Type": "micro details are not offered yet", "Curvature Intensity": "micro details",
         "Height Details Intensity": "micro details", "AO Radius": "micro details", "AO Depth": "micro details"}


class MetalEdgeWear(Generator):
    """Wear on the convex edges (curvature), broken by a grunge (triplanar on the position and world space normal
    maps), kept out of the occluded areas."""
    type = "paint_metal_edge_wear"
    label = "Metal Edge Wear"
    catalog = "painter.generator.metal_edge_wear"
    doc = DOC + "generators/metal-edge-wear"
    includes = ("grunge",)
    inputs = mesh_inputs("curvature", "baked_ambient_occlusion", "world_space_normals", "position")
    params = {"invert": B("Invert", False), "wear_level": F("Wear Level", 0.5, 0.0, 1.0),
              "wear_contrast": F("Wear Contrast", 0.3, 0.0, 1.0), "use_triplanar": B("Use Triplanar", True),
              "triplanar_blending_contrast": F("Triplanar Blending Contrast", 0.5, 0.0, 1.0),
              "grunge_amount": F("Grunge Amount", 0.5, 0.0, 1.0), "grunge_scale": F("Grunge Scale", 1.0, 0.1, 8.0),
              "edges_smoothness": F("Edges Smoothness", 0.3, 0.0, 1.0),
              "ambient_occlusion_masking": F("Ambient Occlusion Masking", 0.5, 0.0, 1.0),
              "curvature_weight": F("Curvature Weight", 0.8, 0.0, 1.0)}
    unsupported = {**SEED, **MICRO}
