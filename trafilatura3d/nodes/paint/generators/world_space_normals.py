from trafilatura3d.nodes.paint._lib import DOC, B, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import (Staged, custom, global_params, group, texture_params,
                                                      three_params)


class WorldSpaceNormals(Staged):
    """Masks by facing direction from the world space normals: Right to Left (facing +Z white), Top To Bottom
    (facing up white), Front to Back (facing +X white), each with an amount, Invert and Blending Mode, combined in
    that order; then the layer's Invert, Blur, Balance, Contrast, Brightness; a custom texture blended over it;
    then the global adjustments."""
    type = "paint_world_space_normals"
    label = "World Space Normals"
    catalog = "painter.generator.world_space_normals"
    doc = DOC + "generators/world-space-normals"
    includes = ("generators/common", "generators/globals")
    templates = (("texture_layer", "texture"),)
    channel_blurs = ("blur", None, None)
    inputs = {**mesh_inputs("world_space_normals", "position"), "texture": custom("Texture")}
    params = {**global_params(), "use_texture": B("Use Texture", False),
              **group({"invert": B("Invert", False), "blur": F("Blur", 0.0, 0.0, 32.0),
                       "balance": F("Balance", 0.5, 0.0, 1.0), "contrast": F("Contrast", 0.0, -1.0, 1.0),
                       "brightness": F("Brightness", 0.0, -1.0, 1.0)}, "World Space Normal"),
              **three_params("", "World Space Normal", "/"),
              **group({"texture_opacity": F("Texture Opacity", 1.0, 0.0, 1.0)}, "Texture"),
              **texture_params("texture_", "Texture")}
