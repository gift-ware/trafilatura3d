from trafilatura3d.nodes.materials._base import F, material_inputs
from trafilatura3d.nodes.materials.weathering._common import (BAKED_AO, BAKED_WSN, MASK, NO_METALLIC, WeatheringNode,
                                                        advanced, blending)


class LeatherWeathering(WeatheringNode):
    """Wear and dirt on a leather material: dust on upward faces, smudges in occluded areas, sharpened
    lighter edges of the relief, a worn leather look (fine cracks of Cracks Scale / Warp) applied
    globally (Used) and in the creases found by the AO (Age, Age Threshold), lightened and
    desaturated by the Used Leather controls, sharp-edge scratches. Per-channel Blending intensities."""

    type = "leather_weathering"
    label = "Leather Weathering"
    catalog = "leather_weathering"
    unsupported = {"Age Threshlod": "the documentation's typo of Age Threshold"}
    inputs = {**material_inputs(), "baked_ambient_occlusion": BAKED_AO, "baked_normal_world_space": BAKED_WSN,
              "mask": MASK}
    params = {
        **advanced(),
        "dust": F(0.0, "Dust", 0.0, 1.0, group="Effect"),
        "dirtiness": F(0.0, "Dirtiness", 0.0, 1.0, group="Effect"),
        "edges_wearing": F(0.5, "Edges Wearing", 0.0, 1.0, group="Effect"),
        "used": F(0.0, "Used", 0.0, 1.0, group="Effect"),
        "age": F(0.0, "Age", 0.0, 1.0, group="Effect"),
        "age_threshold": F(0.5, "Age Threshold", 0.0, 1.0, group="Effect"),
        "cracks_scale": F(5.0, "Cracks Scale", 1.0, 16.0, group="Effect"),
        "cracks_warp_intensity": F(0.72, "Cracks Warp Intensity", 0.0, 1.0, group="Effect"),
        "sharp_edges_scratches_scale": F(1.0, "Sharp Edges Scratches Scale", 1.0, 32.0, group="Effect"),
        "sharp_edges_scratches_warp_intensity": F(0.2, "Sharp Edges Scratches Warp Intensity", 0.0, 1.0, group="Effect"),
        "used_leather_desaturation": F(0.25, "Used Leather Desaturation", 0.0, 1.0, group="Effect"),
        "used_leather_brightness": F(0.7, "Used Leather Brightness", 0.0, 1.0, group="Effect"),
        **blending(NO_METALLIC),
    }
