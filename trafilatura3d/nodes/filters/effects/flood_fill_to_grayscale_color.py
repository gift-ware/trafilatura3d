from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class FloodFillToGrayscaleColor(Effect):
    """Flat swatches per shape: the Grayscale/Color Input sampled at the centre
    of the shape's box (Filtering Mode: nearest texel or bilinear; black without it), then pushed
    towards white (> 0) or black (< 0) by Luminance/Color Adjustment + a random value per shape in
    0-1 x Luminance/Color Random. For color the same per HSL channel
    with HSL Adjustment + an independent random per channel x HSL Random: the hue turns, saturation and
    lightness are pushed towards 1 or 0; alpha is the input's. Grayscale or color follows the input."""

    type = "flood_fill_to_grayscale_color"
    label = "Flood Fill to Grayscale/Color"
    category = "Filters/Effects"
    catalog = "flood_fill_to_grayscale_color"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill": Input(PortType.COLOR, "Flood Fill"),
              "grayscale_color_input": Input(PortType.ANY, "Grayscale/Color Input", default=(0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output()}
    params = {
        "luminance_color_adjustment": Param(ParamKind.FLOAT, 0.0, "Luminance/Color Adjustment", -1.0, 1.0),
        "luminance_color_random": Param(ParamKind.FLOAT, 0.0, "Luminance/Color Random", -1.0, 1.0),
        "hsl_adjustment": Param(ParamKind.FLOAT3, [0.0, 0.0, 0.0], "HSL Adjustment", -1.0, 1.0),
        "hsl_random": Param(ParamKind.FLOAT3, [0.0, 0.0, 0.0], "HSL Random", -1.0, 1.0),
        "filtering_mode": Param(ParamKind.ENUM, "Nearest", "Filtering Mode", choices=("Nearest", "Bilinear")),
    }
