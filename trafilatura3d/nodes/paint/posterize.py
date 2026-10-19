from trafilatura3d.nodes.paint._lib import DOC, B, F, Filter


class Posterize(Filter):
    """Values quantized to Colors Amount levels."""
    type = "paint_posterize"
    label = "Posterize"
    catalog = "painter.filter.posterize"
    doc = DOC + "filters/standard-filters/posterize"
    params = {"colors_amount": F("Colors Amount", 4.0, 2.0, 64.0), "apply_to_alpha": B("Apply to Alpha", False)}
