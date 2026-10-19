from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from .warp import PixelSized


class DirectionalWarp(PixelSized):
    """Sample the input displaced along the Warp angle by (Intensity input - Intensity map offset)
    x Intensity / 256 of the image (Intensity 256 x W moves by W of the image) x Pixel Size per axis,
    as Warp (a Pixel Size of the aspect ratio keeps the move square on a non-square image)."""

    type = "directional_warp"
    label = "Directional warp"
    category = "Atomic nodes"
    catalog = "directional_warp"
    inputs = {"input": Input(label="Input"),
              "intensity_input": Input(PortType.GRAY, "Intensity input", default=(1.0, 1.0, 1.0, 1.0))}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 64.0),
        "warp_angle": Param(ParamKind.FLOAT, 0.0, "Warp angle", 0.0, 1.0),
        "input_filtering_mode": Param(ParamKind.ENUM, "Bilinear", "Input filtering mode",
                                      choices=("Nearest", "Bilinear")),
        "intensity_map_offset": Param(ParamKind.FLOAT, 0.0, "Intensity map offset", 0.0, 1.0),
    }
