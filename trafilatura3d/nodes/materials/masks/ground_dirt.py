from trafilatura3d.nodes.materials._base import F
from trafilatura3d.nodes.materials.masks._common import MASK, POS_GRAY, MaskNode


class GroundDirt(MaskNode):
    """Dirt rising from the ground: full at the bottom of the model (grayscale Position = height),
    fading out at Dirt Height, with a built-in noise for an irregular border."""

    type = "ground_dirt"
    label = "Ground Dirt"
    catalog = "ground_dirt"
    inputs = {"position": POS_GRAY, "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
        "dirt_height": F(0.5, "Dirt Height", 0.0, 1.0),
    }
