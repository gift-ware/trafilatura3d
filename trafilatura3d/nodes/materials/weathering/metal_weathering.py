from trafilatura3d.nodes.materials._base import C, E, F, material_inputs
from trafilatura3d.nodes.materials.weathering._common import (BAKED_AO, BAKED_WSN, CHANNEL_BLENDING, MASK, WeatheringNode,
                                                        advanced, blending)


class MetalWeathering(WeatheringNode):
    """Weathering of a painted metal material: the paint cracks and peels (Paint Peeling, crack
    network of Paint Cracks Scale / Warp) and wears off the edges of the relief (Edges Wearing) and in
    sharp-edge scratches, revealing the raw metal (Raw Metal Color / Specular / Glossiness / Roughness,
    fully metallic); rust or verdigris grows in occluded areas (baked AO) and on the bare metal
    (Rust), flaking (Rust Peeling); dust settles on upward faces, dirt in occluded areas. Normal
    Intensity (0-64) is the strength of the added relief."""

    type = "metal_weathering"
    label = "Metal Weathering"
    catalog = "metal_weathering"
    inputs = {**material_inputs(), "baked_normal_ws": BAKED_WSN, "baked_ambient_occlusion": BAKED_AO, "mask": MASK}
    params = {
        **advanced(),
        "dust": F(0.0, "Dust", 0.0, 1.0, group="Effect"),
        "dirtiness": F(0.0, "Dirtiness", 0.0, 1.0, group="Effect"),
        "edges_wearing": F(0.0, "Edges Wearing", 0.0, 1.0, group="Effect"),
        "paint_peeling": F(0.0, "Paint Peeling", 0.0, 1.0, group="Effect"),
        "rust": F(0.0, "Rust", 0.0, 1.0, group="Effect"),
        "rust_peeling": F(0.5, "Rust Peeling", 0.0, 1.0, group="Effect"),
        "rust_verdigris": E("Rust", "Rust Verdigris", ("Rust", "Verdigris"), group="Effect"),
        "paint_cracks_scale": F(4.0, "Paint Cracks Scale", 1.0, 16.0, group="Effect"),
        "paint_cracks_warp_intensity": F(0.15, "Paint Cracks Warp Intensity", 0.0, 1.0, group="Effect"),
        "sharp_edges_scratches_scale": F(8.0, "Sharp Edges Scratches Scale", 1.0, 32.0, group="Effect"),
        "sharp_edges_scratches_warp_intensity": F(0.05, "Sharp Edges Scratches Warp Intensity", 0.0, 1.0,
                                                  group="Effect"),
        "raw_metal_color": C([0.502, 0.502, 0.502], "Raw Metal Color", group="Effect"),
        "raw_metal_specular_color": C([0.588, 0.588, 0.588], "Raw Metal Specular Color", group="Effect"),
        "raw_metal_glossiness_value": F(0.251, "Raw Metal Glossiness Value", 0.0, 1.0, group="Effect"),
        "raw_metal_roughness_value": F(0.251, "Raw Metal Roughness Value", 0.0, 1.0, group="Effect"),
        **blending(CHANNEL_BLENDING, normal_max=64.0,
                   defaults={"Normal": 5.0, "Ambient Occlusion": 0.5, "Height": 0.1}),
    }
