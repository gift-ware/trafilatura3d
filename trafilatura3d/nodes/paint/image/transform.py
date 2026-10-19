from trafilatura3d.nodes.base import Param, ParamKind
from trafilatura3d.nodes.paint._lib import DOC, B, E, F, Filter


class Transform(Filter):
    """Scales (Scaling x Scaling X / Y: 2 = twice as big), rotates (turns, clockwise) and offsets (UV, Offset +
    Offset X / Y) the input about the centre, after Flip mode mirrors it. Repeat mode: Repeat tiles the input,
    Clamp stretches its border pixels, None leaves the uncovered area empty (transparent black). Tile Safe
    Rotation snaps rotation and overall Scaling to an integer lattice transform (like Safe Transform), so with
    Repeat the result still tiles. Shrinking averages the footprint of each output pixel."""
    type = "paint_transform"
    label = "Transform"
    catalog = "painter.filter.transform"
    doc = DOC + "filters/standard-filters/transform"
    includes = ("image/_image",)
    params = {"repeat_mode": E("Repeat mode", ("Repeat", "Clamp", "None")),
              "flip_mode": E("Flip mode", ("None", "Horizontal", "Vertical", "Both")),
              "scaling_x": F("Scaling X", 1.0, 0.05, 4.0), "scaling_y": F("Scaling Y", 1.0, 0.05, 4.0),
              "offset_x": F("Offset X", 0.0, -1.0, 1.0), "offset_y": F("Offset Y", 0.0, -1.0, 1.0),
              "tile_safe_rotation": B("Tile Safe Rotation", False),
              "rotation": F("Rotation", 0.0, 0.0, 1.0), "scaling": F("Scaling", 1.0, 0.05, 4.0),
              "offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Offset", -1.0, 1.0)}
