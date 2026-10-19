from trafilatura3d.nodes.paint._lib import DOC, B, F, Filter


class HistogramScan(Filter):
    """A window of the histogram (at Position, narrower with Contrast) stretched to 0-1."""
    type = "paint_histogram_scan"
    label = "Histogram Scan"
    catalog = "painter.filter.histogram_scan"
    doc = DOC + "filters/standard-filters/histogram-scan"
    params = {"position": F("Position", 0.5, 0.0, 1.0), "contrast": F("Contrast", 0.0, 0.0, 1.0),
              "invert_position": B("Invert Position", False)}
