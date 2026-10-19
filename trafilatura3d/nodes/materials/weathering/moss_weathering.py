from trafilatura3d.nodes.materials._base import C, F, material_inputs
from trafilatura3d.nodes.materials.weathering._common import (BAKED_HEIGHT, BAKED_POS, MASK, NO_METALLIC, WeatheringNode,
                                                        advanced, blending)


class MossWeathering(WeatheringNode):
    """Overgrown moss on a full material: it settles on the top of the model (baked Position, green
    up) and in the low areas of the additional Height, and Moss Propagation grows it in steps from a
    light dusting of Moss Color to thick, darker, fuzzy moss raised above the surface."""

    type = "moss_weathering"
    label = "Moss Weathering"
    catalog = "moss_weathering"
    inputs = {**material_inputs(), "baked_position": BAKED_POS, "baked_height": BAKED_HEIGHT, "mask": MASK}
    params = {
        **advanced(),
        "moss_propagation": F(1.0, "Moss Propagation", 0.0, 1.0, group="Effect"),
        "moss_color": C([0.439, 0.455, 0.0745], "Moss Color", group="Effect"),
        **blending(NO_METALLIC, defaults={"Normal": 0.25, "Height": 0.5}),
    }
