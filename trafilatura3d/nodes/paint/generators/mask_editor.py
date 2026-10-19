from trafilatura3d.nodes.paint._lib import DOC, B, E, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import (BLEND_MODES, MICRO_INPUTS, Staged, custom, global_params, group,
                                                      micro_params, texture_params, three_params)
from trafilatura3d.nodes.paint.generators.curvature import BAND_LABELS, CURVATURE_MODES, band_params
from trafilatura3d.nodes.paint.generators.position import gradient_params


class MaskEditor(Staged):
    """Layers blended from the bottom up onto black, each at its Opacity with its Blending Mode: custom Texture,
    Texture 2, Ambient Occlusion (+ micro details), Curvature (bands, micro details, Mode), World Space Normal
    (facing gradients), Position Gradient (three gradients blended among themselves first), Thickness (no
    blending mode in the documentation: Copy). Then the global adjustments. Default: the curvature edges."""
    type = "paint_mask_editor"
    label = "Mask Editor"
    catalog = "painter.generator.mask_editor"
    doc = DOC + "generators/mask-editor"
    includes = ("generators/common", "generators/globals", "generators/micro")
    templates = (("texture_layer", "texture"), ("texture_layer", "texture_2"))
    channel_blurs = ("ao_blur", "wsn_blur", "thickness_blur")
    bands = tuple("curvature_" + b.lower() for b in BAND_LABELS)
    inputs = {**mesh_inputs("curvature", "baked_ambient_occlusion", "world_space_normals", "position", "thickness"),
              "texture": custom("Texture"), "texture_2": custom("Texture (Secondary)"), **MICRO_INPUTS}
    params = {
        **global_params(),
        "texture_opacity": F("Texture Opacity", 0.0, 0.0, 1.0),
        "texture_2_opacity": F("Texture 2 Opacity", 0.0, 0.0, 1.0),
        "ambient_occlusion_opacity": F("Ambient Occlusion Opacity", 0.0, 0.0, 1.0),
        "curvature_opacity": F("Curvature Opacity", 1.0, 0.0, 1.0),
        "world_space_normal_opacity": F("World Space Normal Opacity", 0.0, 0.0, 1.0),
        "position_gradient_opacity": F("Position Gradient Opacity", 0.0, 0.0, 1.0),
        "thickness_opacity": F("Thickness Opacity", 0.0, 0.0, 1.0),
        **texture_params("texture_", "Texture", "Copy", non_square=True),
        **texture_params("texture_2_", "Texture 2", "Copy", non_square=True),
        **group({"ao_invert": B("Invert", False), "ao_blending_mode": E("Blending Mode", BLEND_MODES, "Copy"),
                 "ao_blur": F("Blur", 0.0, 0.0, 32.0), "ao_balance": F("Balance", 0.5, 0.0, 1.0),
                 "ao_contrast": F("Contrast", 0.0, -1.0, 1.0)}, "Ambient Occlusion"),
        **group({"curvature_invert": B("Invert", False),
                 "curvature_blending_mode": E("Blending Mode", BLEND_MODES, "Copy"),
                 "curvature_mode": E("Mode", CURVATURE_MODES), **band_params("curvature_"),
                 "curvature_contrast": F("Contrast", 0.0, -1.0, 1.0),
                 "curvature_brightness": F("Brightness", 0.0, -1.0, 1.0)}, "Curvature"),
        **group({"wsn_invert": B("Invert", False), "wsn_blending_mode": E("Blending Mode", BLEND_MODES, "Copy"),
                 "wsn_blur": F("Blur", 0.0, 0.0, 32.0), "wsn_balance": F("Balance", 0.5, 0.0, 1.0),
                 "wsn_contrast": F("Contrast", 0.0, -1.0, 1.0), "wsn_brightness": F("Brightness", 0.0, -1.0, 1.0)},
                "World Space Normal"),
        **three_params("wsn_", "World Space Normal", "/"),
        **gradient_params("position_", "Position Gradient", " - "),
        **group({"position_blending_mode": E("Blending mode", BLEND_MODES, "Copy")}, "Position Gradient"),
        **group({"thickness_invert": B("Invert", False), "thickness_blur": F("Blur", 0.0, 0.0, 32.0),
                 "thickness_contrast": F("Contrast", 0.0, -1.0, 1.0),
                 "thickness_brightness": F("Brightness", 0.0, -1.0, 1.0)}, "Thickness"),
        **micro_params(),
    }
