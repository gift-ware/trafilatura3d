from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class SafeTransform(BasicNode):
    """Tiling-safe Transform 2D: the input is repeated Tile times, with whole copies
    in the image's corners (an even Tile moves the input by half an image); the Offset moves the input,
    snapped down to whole pixels (Random: a per-seed random offset). Tile Safe Rotation (turns,
    clockwise) takes the rotation down to a multiple of 45 degrees and tiles a diagonal one sqrt(2)
    times more, so the result always tiles (45 degrees of Tile 1 = Tile (1, 1)). Symmetry flips the
    input on X, Y or both. Mipmap: Automatic averages the footprint when shrinking, Manual uses
    2^Mipmap Level texels (sharper). Without snapping or tiling, uncovered areas show Background
    Color."""

    type = "safe_transform"
    label = "Safe Transform"
    category = "Filters/Transforms"
    catalog = "safe_transform"
    inputs = {"input": Input(label="Input")}
    params = {
        "tile": Param(ParamKind.INT, 1, "Tile", 1, 16),
        "offset_mode": Param(ParamKind.ENUM, "Manual", "Offset Mode", choices=("Manual", "Random")),
        "offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Offset", 0.0, 1.0),
        "rotation": Param(ParamKind.FLOAT, 0.0, "Rotation", 0.0, 1.0),
        "tile_safe_rotation": Param(ParamKind.BOOL, True, "Tile Safe Rotation"),
        "symmetry": Param(ParamKind.ENUM, "none", "Symmetry", choices=("none", "X", "Y", "X+Y")),
        "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 0.0], "Background Color"),
        "mipmap_mode": Param(ParamKind.ENUM, "Automatic", "Mipmap Mode", choices=("Automatic", "Manual")),
        "mipmap_level": Param(ParamKind.INT, 0, "Mipmap Level", 0, 10),
    }
