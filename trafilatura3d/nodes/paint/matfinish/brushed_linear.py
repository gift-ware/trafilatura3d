from trafilatura3d.nodes.paint._lib import SEED, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, Finish, channels


class MatFinishBrushedLinear(Finish):
    """Linear brushed metal: long thin streaks along one direction (Rotation), Length = how long they run, in the
    normal channel with a faint trace in base color and roughness."""
    type = "paint_metal_finish_brushed_linear"
    label = "Metal Finish Brushed Linear"
    catalog = "painter.filter.matfinish_brushed_linear"
    doc = FINISH_DOC + "matfinish-brushed-linear"
    inputs, outputs = channels("basecolor", "roughness", "normal")
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "brushing_intensity": F("Brushing Intensity", 0.5, 0.0, 1.0),
              "length": F("Length", 0.6, 0.0, 1.0), "rotation": F("Rotation", 0.0, 0.0, 360.0)}
    unsupported = SEED
