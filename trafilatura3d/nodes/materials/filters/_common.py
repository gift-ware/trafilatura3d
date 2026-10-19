from trafilatura3d.nodes.materials._base import CHANNELS, B, C, F

BLEND_MODES = ("Normal", "Add", "Subtract", "Multiply", "Add/Sub", "Max", "Min", "Switch")
# Channels on by default in the material blending nodes (the others' outputs are hidden).
BLEND_ON = ("diffuse", "basecolor", "normal", "specular", "glossiness", "roughness", "metallic", "height")


def group(ch: str) -> str:
    """The parameter group of a channel (its label)."""
    return CHANNELS[ch][0]


def id_mask_params(fuzziness: float = 0.1) -> dict:
    """The Color ID mask switch and its color picker (green, no padding)."""
    return {
        "color_id_mask": B(False, "Color ID Mask"),
        "color": C([0.0, 1.0, 0.0], "Color"),
        "fuzziness": F(fuzziness, "Fuzziness", 0.01, 1.0),
        "padding": F(0.0, "Padding", 0.0, 1.0),
    }
