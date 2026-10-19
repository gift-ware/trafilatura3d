from trafilatura3d.nodes.materials._base import F, material_inputs
from trafilatura3d.nodes.materials.weathering._common import (BAKED_AO, BAKED_CURV, BAKED_WSN, MASK, NO_METALLIC,
                                                        WeatheringNode, advanced, blending)


class RockWeathering(WeatheringNode):
    """Weathering of a rock material from baked AO, Curvature and world space normals: dust on
    upward faces, dirt in occluded areas, chipped lighter convex edges (Edges Wearing), a used,
    faded look (Used Rock, with its desaturation and brightness), carved cracks (Cracks Scale and
    Intensity), darker aged patches in occluded areas (Age, Age Threshold) and sharp-edge scratches.
    Normal Intensity (0-64) is the strength of the added relief; other Blending values weigh channels."""

    type = "rock_weathering"
    label = "Rock Weathering"
    catalog = "rock_weathering"
    unsupported = {"Age Threshlod": "the documentation's typo of Age Threshold"}
    inputs = {**material_inputs(), "baked_ambient_occlusion": BAKED_AO, "baked_curvature": BAKED_CURV,
              "baked_normal_ws": BAKED_WSN, "mask": MASK}
    params = {
        **advanced(),
        "dust": F(0.0, "Dust", 0.0, 1.0, group="Effect"),
        "dirtiness": F(0.0, "Dirtiness", 0.0, 1.0, group="Effect"),
        "edges_wearing": F(0.0, "Edges Wearing", 0.0, 1.0, group="Effect"),
        "used_rock": F(0.0, "Used Rock", 0.0, 1.0, group="Effect"),
        "cracks_scale": F(32.0, "Cracks Scale", 1.0, 60.0, group="Effect"),
        "cracks_intensity": F(0.5, "Cracks Intensity", 0.0, 1.0, group="Effect"),
        "age": F(0.0, "Age", 0.0, 1.0, group="Effect"),
        "age_threshold": F(0.5, "Age Threshold", 0.0, 1.0, group="Effect"),
        "sharp_edges_scratches_scale": F(1.0, "Sharp Edges Scratches Scale", 1.0, 32.0, group="Effect"),
        "sharp_edges_scratches_warp_intensity": F(0.2, "Sharp Edges Scratches Warp Intensity", 0.0, 1.0, group="Effect"),
        "used_rock_desaturation": F(0.0, "Used Rock Desaturation", 0.0, 1.0, group="Effect"),
        "used_rock_brightness": F(0.53, "Used Rock Brightness", 0.0, 1.0, group="Effect"),
        **blending(NO_METALLIC, normal_max=64.0, defaults={"Normal": 5.0}),
    }
