from trafilatura3d.nodes.paint._lib import DOC, B, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import BorderGenerator


class UVBorderDistance(BorderGenerator):
    """White on the UV island borders (computed `uv_islands` map: outside the UVs or another island), fading to
    black at Distance (px at 256) inside the islands; Smoothness goes from a round shoulder (0) to a smooth S fall
    (1); then Balance, Contrast, Invert. Outside the islands: the border's value."""
    type = "paint_uv_border_distance"
    label = "UV Border Distance"
    catalog = "painter.generator.uv_border_distance"
    doc = DOC + "generators/uv-border-distance"
    inputs = mesh_inputs("uv_islands")
    params = {"invert": B("Invert", False), "balance": F("Balance", 0.5, 0.0, 1.0),
              "contrast": F("Contrast", 0.0, -1.0, 1.0), "smoothness": F("Smoothness", 0.5, 0.0, 1.0),
              "distance": F("Distance", 8.0, 0.0, 64.0)}
