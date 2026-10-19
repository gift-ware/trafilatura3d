from dataclasses import replace

from trafilatura3d.nodes.paint._lib import DOC, F, Filter

RANGES = ("Shadows", "Midtones", "Highlights")


class ColorCorrect(Filter):
    """Each tonal range (weights of the luminance: shadows (1-l)^2, midtones 2l(1-l), highlights l^2) gets its own
    Contrast (around the range's center 0.2 / 0.5 / 0.8), Luminosity (added, x0.5) and Saturation (away from the
    luminance); the three results are mixed by the weights. All sliders -1..1, 0 = unchanged."""
    type = "paint_color_correct"
    label = "Color Correct"
    catalog = "painter.filter.color_correct"
    doc = DOC + "filters/standard-filters/color-correct"
    includes = ("color/color",)
    params = {f"{r.lower()}_{name}": replace(F(name.capitalize(), 0.0, -1.0, 1.0), group=r)
              for r in RANGES for name in ("contrast", "luminosity", "saturation")}
