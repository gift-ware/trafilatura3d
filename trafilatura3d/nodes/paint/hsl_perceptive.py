from trafilatura3d.nodes.paint._lib import DOC, E, F, Filter


class HSLPerceptive(Filter):
    """Hue, saturation and lightness shift: Perceptive in OKLCh (perceptual lightness, chroma, hue of the linear
    color), Regular in HSL of the stored values. 0.5 = unchanged."""
    type = "paint_hsl_perceptive"
    label = "HSL Perceptive"
    catalog = "painter.filter.hsl_perceptive"
    doc = DOC + "filters/standard-filters/hsl-perceptive"
    params = {"mode": E("Mode", ("Perceptive", "Regular")), "hue": F("Hue", 0.5, 0.0, 1.0),
              "saturation": F("Saturation", 0.5, 0.0, 1.0), "lightness": F("Lightness", 0.5, 0.0, 1.0)}
