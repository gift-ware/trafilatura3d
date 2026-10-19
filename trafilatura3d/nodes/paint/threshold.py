from trafilatura3d.nodes.paint._lib import DOC, E, F, Filter


class Threshold(Filter):
    """Black or white by comparing the luminance with Threshold."""
    type = "paint_threshold"
    label = "Threshold"
    catalog = "painter.filter.threshold"
    doc = DOC + "filters/standard-filters/threshold"
    params = {"threshold": F("Threshold", 0.5, 0.0, 1.0),
              "mode": E("Mode", ("Greater", "Greater or equal", "Lower", "Lower or equal"))}
    unsupported = {"_mode": "internal value", "_threshold": "internal value", "_threshold_min": "internal value",
                   "_threshold_max": "internal value"}
