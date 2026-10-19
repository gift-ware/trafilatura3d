from trafilatura3d.nodes.paint._lib import DOC, B, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import (Staged, custom, global_params, group, texture_params,
                                                      three_params)


def gradient_params(prefix: str, name: str, sub: str) -> dict:
    """Invert, Balance, Contrast, Brightness of the gradient layer, then its three directions."""
    return {**group({f"{prefix}invert": B("Invert", False), f"{prefix}balance": F("Balance", 0.5, 0.0, 1.0),
                     f"{prefix}contrast": F("Contrast", 0.0, -1.0, 1.0),
                     f"{prefix}brightness": F("Brightness", 0.0, -1.0, 1.0)}, name),
            **three_params(prefix, name, sub)}


class Position(Staged):
    """Gradients across the mesh from the position map: Right to Left (+Z white), Top To Bottom (+Y, the top,
    white), Front to Back (+X white), each with an amount, Invert and Blending Mode, combined in that order
    (Mask Builder's documentation: Front = +X, Right = +Z); then the layer's Invert, Balance, Contrast, Brightness;
    a custom texture blended over it; then the global adjustments."""
    type = "paint_position"
    label = "Position"
    catalog = "painter.generator.position"
    doc = DOC + "generators/position"
    includes = ("generators/common", "generators/globals")
    templates = (("texture_layer", "texture"),)
    inputs = {**mesh_inputs("position", "world_space_normals"), "texture": custom("Texture")}
    params = {**global_params(), "use_texture": B("Use Texture", False),
              **gradient_params("", "Position Gradient", "/"),
              **group({"texture_opacity": F("Texture Opacity", 1.0, 0.0, 1.0)}, "Texture"),
              **texture_params("texture_", "Texture")}
