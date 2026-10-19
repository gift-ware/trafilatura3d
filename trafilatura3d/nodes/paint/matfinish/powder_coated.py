from trafilatura3d.nodes.paint._lib import SEED, B, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, TRIPLANAR, TRIPLANAR_PARAM, Finish, channels, custom


class MatFinishPowderCoated(Finish):
    """Powder coated metal: the soft orange peel of the coat in the normal channel and, Flakes Intensity, small
    metallic flakes caught in it (brighter, smoother, more metallic, each tilted its own way). Use Custom Flake:
    the Custom Flake texture (tiled at Scale) is the flake mask."""
    type = "paint_metal_finish_powder_coated"
    label = "Metal Finish Powder Coated"
    catalog = "painter.filter.matfinish_powder_coated"
    doc = FINISH_DOC + "matfinish-powder-coated"
    includes = TRIPLANAR
    inputs, outputs = channels("basecolor", "roughness", "metallic", "normal",
                               extra={"custom_flake": custom("Custom Flake")}, triplanar=True)
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "flakes_intensity": F("Flakes Intensity", 0.5, 0.0, 1.0),
              "use_custom_flake": B("Use Custom Flake", False), **TRIPLANAR_PARAM}
    unsupported = SEED
