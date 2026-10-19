from trafilatura3d.nodes.materials._base import B, F, gray_in
from trafilatura3d.nodes.materials.masks._common import MASK, MaskNode


class Grease(MaskNode):
    """Skin grease where the baked Thickness is low (nose, ears, lips...): below Thickness
    Threshold the effect appears, mottled by a grunge (built-in or the Noise input)."""

    type = "grease"
    label = "Grease"
    catalog = "grease"
    inputs = {"thickness": gray_in("Thickness", 0.5), "noise": gray_in("Noise", 0.5), "mask": MASK}
    params = {
        "level": F(0.5, "Level", 0.0, 1.0),
        "contrast": F(0.5, "Contrast", 0.0, 1.0),
        "thickness_threshold": F(0.5, "Thickness Threshold", 0.0, 1.0),
        "override_noise": B(False, "Override Noise"),
    }
