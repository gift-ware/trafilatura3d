from trafilatura3d.nodes.paint._lib import DOC, F, Filter


class Sharpen(Filter):
    """Unsharp mask on a 3x3 neighbourhood."""
    type = "paint_sharpen"
    label = "Sharpen"
    catalog = "painter.filter.sharpen"
    doc = DOC + "filters/standard-filters/sharpen"
    params = {"sharpen_intensity": F("Sharpen Intensity", 0.5, 0.0, 4.0)}
