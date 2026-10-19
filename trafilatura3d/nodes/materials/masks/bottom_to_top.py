from trafilatura3d.nodes.materials._base import F, gray_in
from trafilatura3d.nodes.materials.masks._common import MASK, POS, MaskNode


class BottomToTop(MaskNode):
    """White at the bottom of the model fading to black at the top (baked Position, green = up),
    the height optionally broken up by the Roughness variation map."""

    type = "bottom_to_top"
    label = "Bottom To Top"
    catalog = "bottom_to_top"
    inputs = {"position": POS, "roughness": gray_in("Roughness", 0.5), "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.5, "Contrast", 0.0, 1.0),
        "roughness_variation": F(0.0, "Roughness_Variation", 0.0, 1.0),
    }
