from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class MakeItTilePatch(BasicNode):
    """Stamp the input (the patch, centred) on a 2^Octave x 2^Octave grid, one stamp per cell, over
    Background Color; the result always tiles. The patch is Pattern size % of a cell wide and high, each axis
    +- Size Variation %, moved by exactly Disorder cells in a random direction, rotated by Rotation + a random
    0..Rotation Variation (degrees), each channel darkened by a random 0..Color Variation (colour inputs) or the
    value by 0..Luminosity Variation (grayscale); its mask is a bell (1 - 4 r^2)^5 over the patch through Levels
    from 1 - Mask Size to 1 - Mask Size x Mask Precision, Mask Warping wobbles it with a noise, and stamps
    overlap in a random order from the Random Seed."""

    type = "make_tileable_patch"
    label = "Make Tileable Patch"
    category = "Filters/Tiling"
    catalog = "make_it_tile_patch"
    inputs = {"input": Input(label="Input")}
    params = {
        "mask_size": Param(ParamKind.FLOAT, 1.0, "Mask Size", 0.0, 1.0),
        "mask_precision": Param(ParamKind.FLOAT, 0.5, "Mask Precision", 0.0, 1.0),
        "mask_warping": Param(ParamKind.FLOAT, 0.0, "Mask Warping", -100.0, 100.0),
        "pattern_size_width": Param(ParamKind.FLOAT, 200.0, "Pattern size width", 0.0, 1000.0),
        "pattern_size_height": Param(ParamKind.FLOAT, 200.0, "Pattern size height", 0.0, 1000.0),
        "disorder": Param(ParamKind.FLOAT, 0.0, "Disorder", 0.0, 1.0),
        "size_variation": Param(ParamKind.FLOAT, 0.0, "Size Variation", 0.0, 100.0),
        "octave": Param(ParamKind.INT, 3, "Octave", 0, 6),
        "rotation": Param(ParamKind.FLOAT, 0.0, "Rotation", -360.0, 360.0),
        "rotation_variation": Param(ParamKind.FLOAT, 0.0, "Rotation Variation", 0.0, 360.0),
        "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 0.0], "Background Color"),
        "color_variation": Param(ParamKind.FLOAT, 0.0, "Color Variation", 0.0, 1.0),
        "luminosity_variation": Param(ParamKind.FLOAT, 0.0, "Luminosity Variation", 0.0, 1.0),
    }
