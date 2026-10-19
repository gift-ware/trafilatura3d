from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class NonSquareTransform(BasicNode):
    """Transform 2D for non-square canvases: Automatic repeats the input along the long side so
    its pixels stay square (a square input on a 2:1 canvas shows twice across), Manual repeats it Tile times
    along each axis, the short one scaled by the canvas' aspect (square pixels: Tile 2 x 2 on a 2:1 canvas is
    Automatic). Offset moves the result (UV, snapped to whole pixels), Rotation turns it (turns, clockwise,
    in square pixel space; Safe Rotation snaps it down to quarter turns). Off-image areas show Background
    Color on axes that do not tile."""

    type = "non_square_transform"
    label = "Non-Square Transform"
    category = "Filters/Transforms"
    catalog = "non_square_transform"
    inputs = {"input": Input(label="Input")}
    params = {
        "tile_mode": Param(ParamKind.ENUM, "Automatic", "Tile Mode", choices=("Automatic", "Manual")),
        "tile": Param(ParamKind.INT2, [1, 1], "Tile", 1, 16),
        "offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Offset", 0.0, 1.0),
        "rotation": Param(ParamKind.FLOAT, 0.0, "Rotation", 0.0, 1.0),
        "safe_rotation_square_only": Param(ParamKind.BOOL, True, "Safe Rotation (Square Only)"),
        "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 0.0], "Background Color"),
    }
