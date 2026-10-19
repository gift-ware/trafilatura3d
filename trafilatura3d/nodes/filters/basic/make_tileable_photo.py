from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class MakeItTilePhoto(BasicNode):
    """Fix the edges of a non-tiling image: near the left/right edges the image is replaced by a
    copy shifted half a width (whose own seam lies in the middle, hidden), likewise near top/bottom
    with a copy shifted half a height. Mask Size: how far from the edge the copy reaches (fraction
    of half the image); Mask Precision: solid part of that band (1 = a hard edge), as a Levels
    (1 - Size .. 1 - Size x Precision on an edge ramp); Mask Warping:
    a tiling noise wobbles the transition (not the input's blurred luminance).
    The centre of the image is untouched."""

    type = "make_tileable_photo"
    label = "Make Tileable Photo"
    category = "Filters/Tiling"
    catalog = "make_it_tile_photo"
    inputs = {"input": Input(label="Input")}
    params = {
        "mask_warping_h": Param(ParamKind.FLOAT, 0.0, "Mask Warping H", -100.0, 100.0),
        "mask_warping_v": Param(ParamKind.FLOAT, 0.0, "Mask Warping V", -100.0, 100.0),
        "mask_size_h": Param(ParamKind.FLOAT, 0.1, "Mask Size H", 0.0, 1.0),
        "mask_size_v": Param(ParamKind.FLOAT, 0.1, "Mask Size V", 0.0, 1.0),
        "mask_precision_h": Param(ParamKind.FLOAT, 0.5, "Mask Precision H", 0.0, 1.0),
        "mask_precision_v": Param(ParamKind.FLOAT, 0.5, "Mask Precision V", 0.0, 1.0),
    }
