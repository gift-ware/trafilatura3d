from trafilatura3d.nodes.paint._lib import DOC, B, E, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import (MICRO_INPUTS, Staged, custom, global_params, group, micro_params,
                                                      texture_params)

CURVATURE_MODES = ("Edges", "Cavities", "Dual", "Unprocessed")
BAND_LABELS = ("Sharp", "Fine", "Soft", "Medium", "Large", "Big", "Huge")


def band_params(prefix: str = "") -> dict:
    """Strength of the seven bands of curvature detail (1 px to beyond 32 px at 256): 1 keeps the band."""
    return {f"{prefix}{b.lower()}": F(b, 1.0, 0.0, 2.0) for b in BAND_LABELS}


class Curvature(Staged):
    """The baked curvature with its seven bands of detail weighted (Sharp ... Huge), plus micro details (micro normal
    curvature, micro height relief); Mode keeps the edges, the cavities, both or the map as is; Invert, Contrast,
    Brightness; a custom texture blended over it; then the global adjustments."""
    type = "paint_curvature"
    label = "Curvature"
    catalog = "painter.generator.curvature"
    doc = DOC + "generators/curvature"
    includes = ("generators/common", "generators/globals", "generators/micro")
    templates = (("texture_layer", "texture"),)
    bands = tuple(b.lower() for b in BAND_LABELS)
    inputs = {**mesh_inputs("curvature", "world_space_normals", "position"), "texture": custom("Texture"),
              **MICRO_INPUTS}
    params = {**global_params(), "use_texture": B("Use Texture", False), "use_micro_details": B("Use Micro Details", False),
              **group({"invert": B("Invert", False), "mode": E("Mode", CURVATURE_MODES), **band_params(),
                       "contrast": F("Contrast", 0.0, -1.0, 1.0), "brightness": F("Brightness", 0.0, -1.0, 1.0)},
                      "Curvature"),
              **group({"texture_opacity": F("Texture Opacity", 1.0, 0.0, 1.0)}, "Texture"),
              **texture_params("texture_", "Texture"), **micro_params(ao=False, on=True)}
