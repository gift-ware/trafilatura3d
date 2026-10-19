from trafilatura3d.nodes.materials._base import B, C, E, F

MASK_SOURCES = ("None", "Image Average", "Color Parameter", "Input")


def equalizer_params() -> dict:
    return {
        "input_tiled": B(False, "Input Tiled"),
        "radius": F(25.0, "Radius", 0.0, 50.0),
        "bright_dark_balance": F(0.5, "Bright/Dark Balance", 0.0, 1.0),
        "custom_color_variation": B(False, "Custom Color Variation"),
        "hue": F(360.0, "Hue", 0.0, 360.0, group="Color Variation"),
        "chroma": F(1.0, "Chroma", 0.0, 1.0, group="Color Variation"),
        "luma": F(1.0, "Luma", 0.0, 1.0, group="Color Variation"),
        "mask_source": E("None", "Mask Source", MASK_SOURCES, group="Color Variation"),
        "color": C([0.498, 0.498, 0.498], "Color", group="Mask"),
        "hue_range": F(30.0, "Hue Range", 0.0, 360.0, group="Mask"),
        "chroma_range": F(0.5, "Chroma Range", 0.0, 1.0, group="Mask"),
        "luma_range": F(0.5, "Luma Range", 0.0, 1.0, group="Mask"),
        "blur": F(0.0, "Blur", 0.0, 2.0, group="Mask"),
        "smoothness": F(0.0, "Smoothness", 0.0, 2.0, group="Mask"),
    }
