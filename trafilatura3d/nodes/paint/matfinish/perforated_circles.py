from trafilatura3d.nodes.paint._lib import F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, Finish, channels


class MatFinishPerforatedCircles(Finish):
    """Perforated sheet: round holes on a staggered grid (Scale = holes across the UV square, Holes Size = their
    diameter relative to the spacing). The holes are cut in the opacity channel and darkened in base color (so
    they read without an opacity channel); Details Intensity = the pressed rim around each hole in the normal
    channel and its rougher edge."""
    type = "paint_metal_finish_perforated_circles"
    label = "Metal Finish Perforated Circles"
    catalog = "painter.filter.matfinish_perforated_circles"
    doc = FINISH_DOC + "matfinish-perforated-circles"
    inputs, outputs = channels("basecolor", "roughness", "normal", "opacity")
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "holes_size": F("Holes Size", 0.6, 0.0, 1.0),
              "details_intensity": F("Details Intensity", 0.5, 0.0, 1.0)}
