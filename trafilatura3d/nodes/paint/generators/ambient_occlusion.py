from trafilatura3d.nodes.paint._lib import DOC, B, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import (MICRO_INPUTS, Staged, custom, global_params, group, micro_params,
                                                      texture_params)


class AmbientOcclusion(Staged):
    """The baked ambient occlusion (occluded dark), optionally darkened by micro details (micro height occlusion,
    micro normal cavities), with its own Invert, Blur, Balance, Contrast; a custom texture blended over it; then
    the global adjustments."""
    type = "paint_ambient_occlusion"
    label = "Ambient Occlusion"
    catalog = "painter.generator.ambient_occlusion"
    doc = DOC + "generators/ambient-occlusion"
    includes = ("generators/common", "generators/globals", "generators/micro")
    templates = (("texture_layer", "texture"),)
    channel_blurs = ("blur", None, None)
    inputs = {**mesh_inputs("baked_ambient_occlusion", "world_space_normals", "position"),
              "texture": custom("Texture"), **MICRO_INPUTS}
    params = {**global_params(), "use_texture": B("Use Texture", False), "use_micro_details": B("Use Micro Details", False),
              **group({"invert": B("Invert", False), "blur": F("Blur", 0.0, 0.0, 32.0),
                       "balance": F("Balance", 0.5, 0.0, 1.0), "contrast": F("Contrast", 0.0, -1.0, 1.0)},
                      "Ambient Occlusion"),
              **group({"texture_opacity": F("Texture Opacity", 1.0, 0.0, 1.0)}, "Texture"),
              **texture_params("texture_", "Texture"), **micro_params(on=True)}
