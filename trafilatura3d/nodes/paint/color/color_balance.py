from dataclasses import replace

from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, F, Filter

RANGES = ("Highlights", "Midtones", "Shadows")
AXES = (("cyan_red", "Cyan <-> Red"), ("magenta_green", "Magenta <-> Green"), ("yellow_blue", "Yellow <-> Blue"))


class ColorBalance(Filter):
    """Each tonal range (weights of the luminance: shadows (1-l)^2, midtones 2l(1-l), highlights l^2) shifts the
    red, green and blue channels by its three sliders (-1..1, 0 = unchanged; 1 = +0.4 at full weight)."""
    type = "paint_color_balance"
    label = "Color Balance"
    catalog = "painter.filter.color_balance"
    doc = DOC + "filters/standard-filters/color-balance"
    includes = ("color/color",)
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {f"{r.lower()}_{name}": replace(F(label, 0.0, -1.0, 1.0), group=r) for r in RANGES for name, label in AXES}
