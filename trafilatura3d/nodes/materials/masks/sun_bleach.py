from trafilatura3d.nodes.base import Input
from trafilatura3d.nodes.materials._base import COLOR, F
from trafilatura3d.nodes.materials.masks._common import AO, MASK, MaskNode


class SunBleach(MaskNode):
    """Bleaching by sunlight: surfaces facing up (World Space Normal) bleach most, sides less,
    downward faces not at all; occluded areas (AO) are protected by Occlusion. Level moves the
    effect further down the model."""

    type = "sun_bleach"
    label = "Sun Bleach"
    catalog = "sun_bleach"
    inputs = {"normal_world_space": Input(COLOR, "Normal World Space", (0.0, 0.0, 0.0, 0.0)),
              "ambient_occlusion": AO, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "occlusion": F(0.5, "Occlusion", 0.0, 1.0),
    }
