from trafilatura3d.nodes.paint._lib import DOC, F, Filter


class Clamp(Filter):
    """Values limited to [Min, Max]."""
    type = "paint_clamp"
    label = "Clamp"
    catalog = "painter.filter.clamp"
    doc = DOC + "filters/standard-filters/clamp"
    params = {"min": F("Min", 0.0, 0.0, 1.0), "max": F("Max", 1.0, 0.0, 1.0)}
