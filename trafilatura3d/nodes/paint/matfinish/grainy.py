from trafilatura3d.nodes.paint._lib import SEED, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, TRIPLANAR, TRIPLANAR_PARAM, Finish, channels


class MatFinishGrainy(Finish):
    """Grainy metal (sand blasted, cast): a fine isotropic grain with small pits, in the normal channel and as
    a speckle of roughness and base color."""
    type = "paint_metal_finish_grainy"
    label = "Metal Finish Grainy"
    catalog = "painter.filter.matfinish_grainy"
    doc = FINISH_DOC + "matfinish-grainy"
    includes = TRIPLANAR
    inputs, outputs = channels("basecolor", "roughness", "normal", triplanar=True)
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "grain_intensity": F("Grain Intensity", 0.5, 0.0, 1.0),
              **TRIPLANAR_PARAM}
    unsupported = SEED
