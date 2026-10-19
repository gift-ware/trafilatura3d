from trafilatura3d.nodes.paint._lib import DOC, F, Filter


class HistogramShift(Filter):
    """Values shifted by Position, wrapping around 1."""
    type = "paint_histogram_shift"
    label = "Histogram Shift"
    catalog = "painter.filter.histogram_shift"
    doc = DOC + "filters/standard-filters/histogram-shift"
    params = {"position": F("Position", 0.0, 0.0, 1.0)}
