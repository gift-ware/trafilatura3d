from trafilatura3d.nodes.materials._base import F, material_inputs
from trafilatura3d.nodes.materials.weathering._common import (BAKED_AO, BAKED_WSN, MASK, NO_METALLIC, WeatheringNode,
                                                        advanced, blending)


class FabricWeathering(WeatheringNode):
    """Wear and dirt on a fabric material: dust on upward faces (baked world space normal), smudges
    in occluded areas (Dirtiness), dark crease dirt (Used), sharpened and lightened edges of the
    material relief (Edges Wearing), a global tiling wear with creases (Age, limited by the AO
    through Age Threshold, desaturated and brightened by the Old Fabric controls), and sharp-edge
    scratches that scrape Used and Age away. Each channel takes the effect at its Blending intensity."""

    type = "fabric_weathering"
    label = "Fabric Weathering"
    catalog = "fabric_weathering"
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
        "age_threshold": F(1.0, "Age Threshold", 0.0, 1.0, group="Effect"),
        "age_creases": F(0.5, "Age Creases", 0.0, 1.0, group="Effect"),
        "sharp_edges_scratches_scale": F(1.0, "Sharp Edges Scratches Scale", 1.0, 32.0, group="Effect"),
        "sharp_edges_scratches_warp_intensity": F(0.2, "Sharp Edges Scratches Warp Intensity", 0.0, 1.0, group="Effect"),
        "old_fabric_desaturation": F(0.25, "Old Fabric Desaturation", 0.0, 1.0, group="Effect"),
        "old_fabric_brightness": F(0.55, "Old Fabric Brightness", 0.0, 1.0, group="Effect"),
        **blending(NO_METALLIC),
    }
