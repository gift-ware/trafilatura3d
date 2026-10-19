from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, E, PaintNode

# Thresholds on the base color's brightness (sRGB 0-255 of the linear luminance), from the PBR guide's ranges:
# non-metal albedo no darker than 30 (tolerant) / 50 (strict) sRGB; metal reflectance 70-100% = 180-255 sRGB
# (strict), 60-100% = 155-255 sRGB (tolerant).
ALBEDO = ("Tolerant (30 sRGB)", "Strict (50 sRGB)")
METAL = ("Strict (180-255 sRGB)", "Tolerant (155-255 sRGB)")


class PBRValidate(PaintNode):
    """Validation of the material: non-metal pixels (metallic < 0.5) whose base color is darker than the albedo
    threshold, and metal pixels (metallic >= 0.5) whose base color is darker than the reflectance range, are
    invalid. Writes the base color channel: red = invalid, green = valid, black = not checked by the Validation
    Mode; with Overlay Map the colors are laid at 50% over the base color (unchecked pixels show it as is)."""
    type = "paint_pbr_validate"
    label = "PBR Validate"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.pbr_validate"
    doc = DOC + "filters/advanced-filters/pbr-validate"
    inputs = {"basecolor": Input(PortType.COLOR, "Base Color", (0.5, 0.5, 0.5, 1.0)),
              "metallic": Input(PortType.GRAY, "Metallic", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"basecolor": Output(PortType.COLOR, "Base Color")}
    params = {"validation_mode": E("Validation Mode", ("Albedo", "Metal Reflectance", "Albedo and Metal Reflectance"),
                                   "Albedo and Metal Reflectance"),
              "albedo_dark_range_threshold": E("Albedo Dark Range Threshold", ALBEDO),
              "metal_reflectance_range": E("Metal Reflectance Range", METAL),
              "overlay_map": B("Overlay Map", True)}
