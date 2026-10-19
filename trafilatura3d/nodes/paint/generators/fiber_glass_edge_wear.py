from trafilatura3d.nodes.paint._lib import DOC, SEED, B, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import GenNode, MICRO_INPUTS, custom, micro_params


class FiberGlassEdgeWear(GenNode):
    """Wear on the convex edges (curvature + micro details) broken by a grunge (fbm, triplanar on the position and
    world space normal maps, or the custom grunge), frayed along its border by fibers (thin ridged noise in three
    random directions), kept out of the occluded areas (AO + micro occlusion). Invert inverts the curvature and
    ambient occlusion before they are combined (documentation)."""
    type = "paint_fiber_glass_edge_wear"
    label = "Fiber Glass Edge Wear"
    catalog = "painter.generator.fiber_glass_edge_wear"
    doc = DOC + "generators/fiber-glass-edge-wear"
    includes = ("grunge", "generators/common", "generators/micro")
    templates = (("texture_layer", "custom_grunge"),)
    inputs = {**mesh_inputs("curvature", "baked_ambient_occlusion", "world_space_normals", "position"),
              "custom_grunge": custom("Custom grunge", gray=True), **MICRO_INPUTS}
    params = {"invert": B("Invert", False), "wear_level": F("Wear Level", 0.5, 0.0, 1.0),
              "wear_contrast": F("Wear Contrast", 0.3, 0.0, 1.0), "use_triplanar": B("Use Triplanar", True),
              "triplanar_blending_contrast": F("Triplanar Blending Contrast", 0.5, 0.0, 1.0),
              "grunge_amount": F("Grunge Amount", 0.5, 0.0, 1.0), "use_custom_grunge": B("Use Custom Grunge", False),
              "edges_smoothness": F("Edges Smoothness", 0.3, 0.0, 1.0),
              "ambient_occlusion_masking": F("Ambient Occlusion Masking", 0.5, 0.0, 1.0),
              "curvature_weight": F("Curvature Weight", 0.8, 0.0, 1.0), **micro_params()}
    unsupported = {**SEED}
