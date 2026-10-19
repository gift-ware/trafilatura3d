from trafilatura3d.nodes.materials._base import F, I, material_inputs
from trafilatura3d.nodes.materials.weathering._common import (BAKED_AO, BAKED_CURV, BAKED_POS, CHANNEL_BLENDING, MASK,
                                                        WeatheringNode, advanced, blending)


class RustWeathering(WeatheringNode):
    """Rust on a full material: it spreads (Rust Spreading, edges softened by Spreading Smoothness)
    from occluded areas and edges (baked AO and Curvature) through damage in the varnish (cells of
    Varnish Damage Scale), and drips down the model (baked Position, green up) over Drips Samples
    Amount steps, Drips Intensity long, Drips Smoothness soft. The rust layer is rough, non-metallic,
    slightly raised. Normal Intensity (0-32) is the strength of its relief."""

    type = "rust_weathering"
    label = "Rust Weathering"
    catalog = "rust_weathering"
    unsupported = {"Vernish Damage Scale": "the documentation's typo of Varnish Damage Scale"}
    inputs = {**material_inputs(), "baked_ambient_occlusion": BAKED_AO, "baked_curvature": BAKED_CURV,
              "baked_position": BAKED_POS, "mask": MASK}
    params = {
        **advanced(),
        "rust_spreading": F(0.11, "Rust Spreading", 0.0, 1.0, group="Effect"),
        "spreading_smoothness": F(0.25, "Spreading Smoothness", 0.0, 1.0, group="Effect"),
        "varnish_damage_scale": F(0.3, "Varnish Damage Scale", 0.0, 1.0, group="Effect"),
        "drips_intensity": F(0.2, "Drips Intensity", 0.0, 1.0, group="Effect"),
        "drips_samples_amount": I(8, "Drips Samples Amount", 0, 32, group="Effect"),
        "drips_smoothness": F(0.2, "Drips Smoothness", 0.0, 1.0, group="Effect"),
        **blending(CHANNEL_BLENDING, normal_max=32.0, defaults={"Normal": 6.0, "Height": 0.05}),
    }
