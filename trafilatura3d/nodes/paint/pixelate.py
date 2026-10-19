from trafilatura3d.nodes.paint._lib import DOC, B, F, Filter


class Pixelate(Filter):
    """Pixels Amount square cells across the texture, each with the value at its center."""
    type = "paint_pixelate"
    label = "Pixelate"
    catalog = "painter.filter.pixelate"
    doc = DOC + "filters/standard-filters/pixelate"
    params = {"pixels_amount": F("Pixels Amount", 64.0, 2.0, 1024.0), "apply_to_alpha": B("Apply To Alpha", False)}
