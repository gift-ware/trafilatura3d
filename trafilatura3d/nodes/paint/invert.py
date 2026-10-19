from trafilatura3d.nodes.paint._lib import DOC, B, Filter


class Invert(Filter):
    """1 - value when Invert is on."""
    type = "paint_invert"
    label = "Invert"
    catalog = "painter.filter.invert"
    doc = DOC + "filters/standard-filters/invert"
    params = {"invert": B("Invert", True)}
