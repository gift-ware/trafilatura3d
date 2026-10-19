from trafilatura3d.nodes.base import Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, E, F, Filter

PRESETS = ("Linear", "Smooth", "Ease In", "Ease Out", "Bell", "Valley", "Double Bell", "Steps")


class GradientCurve(Filter):
    """Grayscale remap (luminance of a color input) through a preset curve: Linear x, Smooth (smoothstep), Ease In
    x^2.5, Ease Out 1-(1-x)^2.5, Bell (peak at 0.5), Valley (1 - Bell), Double Bell (peaks at 0.25 and 0.75),
    Steps (4 soft steps); then Balance (0.5 = unchanged), Contrast (0 = unchanged), Invert."""
    type = "paint_gradient_curve"
    label = "Gradient Curve"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.gradient_curve"
    doc = DOC + "filters/advanced-filters/gradient-curve"
    includes = ("color/color",)
    outputs = {"output": Output(PortType.GRAY)}
    params = {"preset": E("Preset", PRESETS, "Smooth"), "invert": B("Invert", False),
              "balance": F("Balance", 0.5, 0.0, 1.0), "contrast": F("Contrast", 0.0, 0.0, 1.0)}
