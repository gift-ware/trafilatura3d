from trafilatura3d.nodes.paint._lib import SEED, B, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, TRIPLANAR, TRIPLANAR_PARAM, Finish, channels, custom


class MatFinishGalvanized(Finish):
    """Galvanized (hot-dip zinc) metal: spangles, large crystals each with its own tilt, brightness and roughness,
    over a fine grain; Flakes Intensity = how much the crystals differ. Use Custom Flake: the Custom Flake texture
    (tiled at Scale) gives each point its value instead of the crystals."""
    type = "paint_metal_finish_galvanized"
    label = "Metal Finish Galvanized"
    catalog = "painter.filter.matfinish_galvanized"
    doc = FINISH_DOC + "matfinish-galvanized"
    includes = TRIPLANAR
    inputs, outputs = channels("basecolor", "roughness", "normal", extra={"custom_flake": custom("Custom Flake")},
                               triplanar=True)
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "flakes_intensity": F("Flakes Intensity", 0.5, 0.0, 1.0),
              "use_custom_flake": B("Use Custom Flake", False), **TRIPLANAR_PARAM}
    unsupported = SEED
