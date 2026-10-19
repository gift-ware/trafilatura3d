from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, F, Filter


class HeightAdjust(Filter):
    """Height offset (added), then multiplied, optionally inverted (signed: the height channel is signed)."""
    type = "paint_height_adjust"
    label = "Height Adjust"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.height_adjustments"
    doc = DOC + "filters/advanced-filters/height-adjustments"
    params = {"invert": B("Invert", False), "offset": F("Offset", 0.0, -1.0, 1.0),
              "multiply": F("Multiply", 1.0, 0.0, 4.0)}
