from trafilatura3d.nodes.paint._lib import SEED, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, TRIPLANAR, TRIPLANAR_PARAM, Finish, channels


class MatFinishGrinded(Finish):
    """Grinded metal: patches of short straight grinding marks, each patch in its own direction (three crossing
    layers broken by a noise), the patches slightly brighter or rougher than each other."""
    type = "paint_metal_finish_grinded"
    label = "Metal Finish Grinded"
    catalog = "painter.filter.matfinish_grinded"
    doc = FINISH_DOC + "matfinish-grinded"
    includes = TRIPLANAR
    inputs, outputs = channels("basecolor", "roughness", "normal", triplanar=True)
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "grinding_intensity": F("Grinding Intensity", 0.5, 0.0, 1.0),
              **TRIPLANAR_PARAM}
    unsupported = SEED
