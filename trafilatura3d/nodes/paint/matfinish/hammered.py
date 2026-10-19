from trafilatura3d.nodes.paint._lib import SEED, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, TRIPLANAR, TRIPLANAR_PARAM, Finish, channels


class MatFinishHammered(Finish):
    """Hammered metal: round dents side by side (bowls of packed cells, slightly oval along Rotation), a ridge
    between them; the dents' bottoms a little smoother. Details Intensity = depth of the dents."""
    type = "paint_metal_finish_hammered"
    label = "Metal Finish Hammered"
    catalog = "painter.filter.matfinish_hammered"
    doc = FINISH_DOC + "matfinish-hammered"
    includes = TRIPLANAR
    inputs, outputs = channels("basecolor", "roughness", "normal", triplanar=True)
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "details_intensity": F("Details Intensity", 0.5, 0.0, 1.0),
              **TRIPLANAR_PARAM, "rotation": F("Rotation", 0.0, 0.0, 360.0)}
    unsupported = SEED
