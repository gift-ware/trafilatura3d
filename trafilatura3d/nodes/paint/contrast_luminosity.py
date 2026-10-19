from trafilatura3d.nodes.paint._lib import DOC, F, Filter


class ContrastLuminosity(Filter):
    """Contrast around 0.5 (positive: steeper, negative: flatter), then Luminosity added."""
    type = "paint_contrast_luminosity"
    label = "Contrast Luminosity"
    catalog = "painter.filter.contrast_luminosity"
    doc = DOC + "filters/standard-filters/contrast-luminosity"
    params = {"contrast": F("Contrast", 0.0, -1.0, 1.0), "luminosity": F("Luminosity", 0.0, -1.0, 1.0)}
