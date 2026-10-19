from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._fx import Effect


class ExtendShape(Effect):
    """Stretches one line (section) of the input across a band, pushing the rest of the image aside.
    Bidirectional: the section through the centre + Extension Position (along the extension
    direction), at Extension Angle (turns, 0 = a vertical section), is smeared over Extension
    Distance (UV), half on each side; Unidirectional: all of it on one side; Start / End Positions:
    the section through Start Position, perpendicular to the Start-End vector, smeared up to End
    Position. Grayscale input: the area before the band gets Start Luminance Offset, the area after
    it End Luminance Offset, interpolated across the band (only on non-black pixels if asked).
    Show Helper draws the band's edges and arrows along the extension."""

    type = "extend_shape"
    label = "Extend Shape"
    category = "Filters/Effects"
    catalog = "extend_shape"
    inputs = {"input": Input(label="Input")}
    params = {
        "mode": Param(ParamKind.ENUM, "Bidirectional", "Mode",
                      choices=("Bidirectional", "Unidirectional", "Start / End Positions")),
        "extension_distance": Param(ParamKind.FLOAT, 0.5, "Extension Distance", 0.0, 1.0),
        "extension_position": Param(ParamKind.FLOAT, 0.0, "Extension Position", -0.5, 0.5),
        "extension_angle": Param(ParamKind.FLOAT, 0.0, "Extension Angle", 0.0, 1.0),
        "start_position": Param(ParamKind.FLOAT2, [0.5, 0.5], "Start Position", 0.0, 1.0),
        "end_position": Param(ParamKind.FLOAT2, [0.6, 0.4], "End Position", 0.0, 1.0),
        "start_luminance_offset": Param(ParamKind.FLOAT, 0.0, "Start Luminance Offset", -1.0, 1.0),
        "end_luminance_offset": Param(ParamKind.FLOAT, 0.0, "End Luminance Offset", -1.0, 1.0),
        "lum_offset_ignores_black_pixels": Param(ParamKind.BOOL, True, "Lum. Offset Ignores Black Pixels"),
        "filtering_mode": Param(ParamKind.ENUM, "Bilinear", "Filtering Mode", choices=("Nearest", "Bilinear")),
        "show_helper": Param(ParamKind.BOOL, False, "Show Helper"),
    }
