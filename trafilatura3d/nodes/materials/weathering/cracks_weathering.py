from trafilatura3d.nodes.materials._base import F, material_inputs
from trafilatura3d.nodes.materials.weathering._common import (BAKED_CURV, BAKED_HEIGHT, MASK, NO_METALLIC, WeatheringNode,
                                                        advanced, blending)


class CracksWeathering(WeatheringNode):
    """Adds a random crack network to a full material: cracks start in concave and low areas (baked
    Curvature and Height) and spread with Cracks Propagation; Cracks Depth carves them into the height
    (and a little into the look: darker, rougher, occluded), the normal follows the carved relief.
    Each channel takes the effect at its Blending intensity."""

    type = "cracks_weathering"
    label = "Cracks Weathering"
    catalog = "cracks_weathering"
    inputs = {**material_inputs(), "baked_curvature": BAKED_CURV, "baked_height": BAKED_HEIGHT, "mask": MASK}
    params = {
        **advanced(),
        "cracks_propagation": F(1.0, "Cracks Propagation", 0.0, 1.0, group="Effect"),
        "cracks_depth": F(0.5, "Cracks Depth", 0.0, 1.0, group="Effect"),
        **blending(NO_METALLIC, defaults={"Normal": 0.3, "Roughness": 0.33, "Height": 0.5}),
    }
