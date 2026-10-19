from trafilatura3d.nodes.paint._lib import DOC, F, Filter


class Smoothstep(Filter):
    """Smooth ramp from Minimum (0) to Maximum (1)."""
    type = "paint_smoothstep"
    label = "Smoothstep"
    catalog = "painter.filter.smoothstep"
    doc = DOC + "filters/standard-filters/smoothstep"
    params = {"minimum": F("Minimum", 0.25, 0.0, 1.0), "maximum": F("Maximum", 0.75, 0.0, 1.0)}
