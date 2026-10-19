from trafilatura3d.nodes.paint._lib import SEED, B, F
from trafilatura3d.nodes.paint.matfinish._finish import FINISH_DOC, TRIPLANAR, TRIPLANAR_PARAM, Finish, channels, custom


class MatFinishRough(Finish):
    """Rough metal: a grunge (Scale Grunge; or the Custom Grunge texture) darkens, roughens and dulls the metal,
    over coarse brushing marks bent by Brushing Warp and a fine flake grain (or the Custom Flake texture)."""
    type = "paint_metal_finish_rough"
    label = "Metal Finish Rough"
    catalog = "painter.filter.matfinish_rough"
    doc = FINISH_DOC + "matfinish-rough"
    includes = TRIPLANAR
    inputs, outputs = channels("basecolor", "roughness", "metallic", "normal",
                               extra={"custom_grunge": custom("Custom Grunge"), "custom_flake": custom("Custom Flake")},
                               triplanar=True)
    params = {"scale": F("Scale", 1.0, 0.25, 8.0), "scale_grunge": F("Scale Grunge", 1.0, 0.25, 8.0),
              "brushing_intensity": F("Brushing Intensity", 0.5, 0.0, 1.0),
              "brushing_warp": F("Brushing Warp", 0.3, 0.0, 1.0), "use_custom_flake": B("Use Custom Flake", False),
              **TRIPLANAR_PARAM, "use_custom_grunge": B("Use Custom Grunge", False)}
    unsupported = SEED
