from trafilatura3d.nodes.paint._lib import SEED, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, TRIPLANAR, TRIPLANAR_PARAM, Finish, channels


class MatFinishRaw(Finish):
    """Raw (unfinished, mill) metal: blotches of darker, rougher, less metallic scale over the metal, and a fine
    uneven surface in the normal channel. Details Intensity = how marked the blotches and the relief are."""
    type = "paint_metal_finish_raw"
    label = "Metal Finish Raw"
    catalog = "painter.filter.matfinish_raw"
    doc = FINISH_DOC + "matfinish-raw"
    includes = TRIPLANAR
    inputs, outputs = channels("basecolor", "roughness", "metallic", "normal", triplanar=True)
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "details_intensity": F("Details Intensity", 0.5, 0.0, 1.0),
              **TRIPLANAR_PARAM}
    unsupported = SEED
