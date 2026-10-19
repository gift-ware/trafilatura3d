from trafilatura3d.nodes.paint._lib import DOC, SEED, B, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import GenNode, MICRO_INPUTS, custom, group, micro_params


class MaskBuilder(GenNode):
    """Legacy multi-purpose mask: grunge (fbm, triplanar on the position and world space normal maps, or the custom
    grunges), ambient occlusion, curvature (convex and concave ranges), a top/down gradient (position), world space
    normal directions, scratches (random line segments) and a scatter of stamps (the Scatter input, or discs),
    each weighted and summed; scratches show only inside the grunge, AO and curvature part. Then Level, Contrast,
    Invert. Directions: Front = +X, Right = +Z (documentation)."""
    type = "paint_mask_builder"
    label = "Mask Builder"
    catalog = "painter.generator.mask_builder"
    doc = DOC + "generators/mask-builder"
    includes = ("grunge", "generators/common", "generators/micro")
    templates = (("texture_layer", "custom_grunge_1"), ("texture_layer", "custom_grunge_2"))
    inputs = {**mesh_inputs("curvature", "baked_ambient_occlusion", "world_space_normals", "position"),
              "custom_grunge_1": custom("Custom grunge 1", gray=True), "custom_grunge_2": custom("Custom grunge 2", gray=True),
              "scatter_input": custom("Scatter input", gray=True), **MICRO_INPUTS}
    params = {
        "level": F("Level", 0.5, 0.0, 1.0), "contrast": F("Contrast", 0.0, -1.0, 1.0), "invert": B("Invert", False),
        "use_triplanar": B("Use Triplanar", True),
        "triplanar_blending_contrast": F("Triplanar Blending Contrast", 0.5, 0.0, 1.0),
        "grunge": F("Grunge", 0.5, 0.0, 1.0), "ao": F("AO", 0.0, 0.0, 1.0), "curvature": F("Curvature", 1.0, 0.0, 1.0),
        "top_down_gradient": F("Top/Down Gradient", 0.0, 0.0, 1.0),
        "world_space_normal": F("World Space Normal", 0.0, 0.0, 1.0),
        "scratches": F("Scratches", 0.0, 0.0, 1.0), "scatter": F("Scatter", 0.0, 0.0, 1.0),
        **group({"grunge_scale": F("Scale", 1.0, 0.1, 8.0), "use_custom_grunge": B("Use Custom Grunge", False),
                 "secondary_custom_grunge": F("Secondary Custom Grunge", 0.0, 0.0, 1.0),
                 "grunge_invert": B("Invert", False)}, "Grunge"),
        **group({"ao_range": F("Range", 0.5, 0.0, 1.0), "ao_contrast": F("Contrast", 0.0, -1.0, 1.0),
                 "ao_noisiness": F("Noisiness", 0.0, 0.0, 1.0), "ao_invert": B("Invert", False)}, "Ambient Occlusion"),
        **group({"convex_range": F("Convex Range", 0.5, 0.0, 1.0), "convex_contrast": F("Convex Contrast", 0.0, -1.0, 1.0),
                 "convex_invert": B("Convex Invert", False), "concave_range": F("Concave Range", 0.0, 0.0, 1.0),
                 "concave_contrast": F("Concave Contrast", 0.0, -1.0, 1.0),
                 "concave_invert": B("Concave Invert", False), "smoothness": F("Smoothness", 0.3, 0.0, 1.0),
                 "level_boost": F("Level Boost", 1.0, 0.0, 4.0), "curvature_noisiness": F("Noisiness", 0.0, 0.0, 1.0)},
                "Curvature"),
        **group({"gradient_range": F("Range", 0.5, 0.0, 1.0), "gradient_contrast": F("Contrast", 0.0, -1.0, 1.0),
                 "gradient_invert": B("Invert", False)}, "Gradient"),
        **group({"top_intensity": F("Top Intensity", 1.0, 0.0, 1.0), "bottom_intensity": F("Bottom Intensity", 0.0, 0.0, 1.0),
                 "front_intensity": F("Front Intensity", 0.0, 0.0, 1.0), "back_intensity": F("Back Intensity", 0.0, 0.0, 1.0),
                 "right_intensity": F("Right Intensity", 0.0, 0.0, 1.0), "left_intensity": F("Left Intensity", 0.0, 0.0, 1.0)},
                "World Space Normal"),
        **group({"scratches_amount": F("Amount", 0.5, 0.0, 1.0), "scratches_scale": F("Scale", 1.0, 0.1, 8.0)}, "Scratches"),
        **group({"scatter_scale": F("Scale", 1.0, 0.1, 8.0), "scatter_density": F("Density", 0.5, 0.0, 1.0),
                 "scatter_size": F("Size", 0.5, 0.0, 1.0), "size_variation": F("Size Variation", 0.3, 0.0, 1.0),
                 "opacity_variation": F("Opacity Variation", 0.3, 0.0, 1.0)}, "Scatter"),
        **micro_params(),
    }
    unsupported = {**SEED}
