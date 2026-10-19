from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import B, E, F, I, MatFX, stage

CHANNELS = ("basecolor", "roughness", "normal", "height")
CC = "Input Color Correction"
ST = "Strokes"
# Tone layers, painted in this order: (group, suffix, grid subdivision, strokes size, has "Strokes Size From Details").
LAYERS = (("Background", "background", 8, 1.2, False), ("Shadows", "shadows", 14, 1.0, True),
          ("Midtones", "midtones", 22, 0.9, True), ("Highlights", "highlights", 32, 0.8, True))


def _layer_params():
    out = {}
    for group, s, grid, size, details in LAYERS:
        out[f"enabled_{s}"] = B("Enabled", True, group)
        out[f"grid_subdivision_{s}"] = I("Grid Subdivision", grid, 1, 128, group)
        out[f"strokes_size_{s}"] = F("Strokes Size", size, 0.1, 3.0, group)
        if details:
            out[f"strokes_size_from_details_{s}"] = F("Strokes Size From Details", 0.5, group=group)
        out[f"warp_intensity_{s}"] = F("Warp Intensity", 0.3, group=group)
    return out


class OilPaint(MatFX):
    """Oil paint look: the source (the base color, or the `image_input` with Input Type Image Input, corrected by
    the Input Color Correction group) is repainted with brush strokes in four tone layers, from the Background
    (everywhere) to the Shadows, Midtones and Highlights (strokes placed where the source has that tone). Each
    layer scatters one stroke per cell of its Grid Subdivision (divided by Global Scale Multiplier): a tapered
    stroke of the source's color at its start (Random Color), Strokes Size x Size long (Scale Random, Size
    Random, smaller on details with Strokes Size From Details), wobbling with Warp Intensity, edges as hard as
    Strokes Hardness, oriented along the source's edges (structure tensor smoothed by Orientation Smoothness,
    Automatic Orientation Multiplier). Fine Details and Sharpness bring back detail, Contour darkens the strokes'
    edges, the Canvas weave (Fibers Count) shows through. Strokes and bristles make the height and the normal
    (Normal Intensity, Normal Strokes Direction Influence: the bristle grooves along the strokes) and vary the
    roughness (Roughness, Roughness Variation per stroke, Roughness Details from the bristles); Effect Intensity
    mixes it all over the material."""
    type = "paint_material_fx_oil_paint"
    label = "Material FX Oil Paint"
    catalog = "painter.filter.matfx_oil_paint"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-oil-paint"
    inputs = {**material_inputs(CHANNELS), "image_input": Input(PortType.COLOR, "Image Input", (0.5, 0.5, 0.5, 1.0))}
    outputs = material_outputs(CHANNELS)
    params = {
        "input_type": E("Input Type", ("Base Color", "Image Input")),
        "effect_intensity": F("Effect Intensity", 1.0), "fine_details": F("Fine Details", 0.25),
        "sharpness": F("Sharpness", 0.3), "roughness": F("Roughness", 0.4),
        "roughness_variation": F("Roughness Variation", 0.2), "roughness_details": F("Roughness Details", 0.3),
        "normal_intensity": F("Normal Intensity", 1.0, 0.0, 4.0),
        "normal_strokes_direction_influence": F("Normal Strokes Direction Influence", 0.5),
        "input_image_contrast": F("Input Image Contrast", 0.0, -1.0, 1.0, CC),
        "input_image_hue": F("Input Image Hue", 0.0, -0.5, 0.5, CC),
        "input_image_saturation": F("Input Image Saturation", 0.5, group=CC),
        "input_image_luminosity": F("Input Image Luminosity", 0.5, group=CC),
        "global_scale_multiplier": F("Global Scale Multiplier", 1.0, 0.25, 4.0, ST),
        "scale_random": F("Scale Random", 0.3, group=ST), "size": F("Size", 1.0, 0.25, 2.0, ST),
        "size_random": F("Size Random", 0.3, group=ST), "random_color": F("Random Color", 0.15, group=ST),
        "strokes_hardness": F("Strokes Hardness", 0.6, group=ST),
        "automatic_orientation_multiplier": F("Automatic Orientation Multiplier", 1.0, group=ST),
        "orientation_smoothness": F("Orientation Smoothness", 3.0, 0.0, 16.0, ST),
        **_layer_params(),
        "amount": F("Amount", 0.3, group="Contour"), "thickness": F("Thickness", 0.3, group="Contour"),
        "texture_intensity": F("Texture Intensity", 0.3, group="Canvas"),
        "fibers_count": I("Fibers Count", 160, 16, 512, "Canvas"),
    }
    unsupported = {}

    def encode(self, ctx, job):
        v = job.values
        src, tensor = ctx.scratch(job, False), ctx.scratch(job, False)
        ctx.dispatch(PREP, job, v, dsts={"src": src, "tensor": tensor})
        tb = soft_blur(ctx, job, tensor, False, max(v["orientation_smoothness"], 0.5))
        strokes, aux = ctx.scratch(job, False), ctx.scratch(job, False)
        ctx.dispatch(STROKES, job, v, srcs={"src": (src, False), "tensor": (tb, False)},
                     dsts={"strokes": strokes, "aux": aux})
        ctx.dispatch(FINAL, job, v, srcs={"src": (src, False), "strokes": (strokes, False), "aux": (aux, False)})


P = OilPaint.params
COLOR_IN = Input(PortType.COLOR)
PREP = stage("paint_material_fx_oil_paint_prep", "oil_paint_prep.wgsl",
             {"basecolor": OilPaint.inputs["basecolor"], "image_input": OilPaint.inputs["image_input"]},
             {"src": Output(PortType.COLOR), "tensor": Output(PortType.COLOR)},
             {k: v for k, v in P.items() if k == "input_type" or k.startswith("input_image_")})
STROKES = stage("paint_material_fx_oil_paint_strokes", "oil_paint_strokes.wgsl", {"src": COLOR_IN, "tensor": COLOR_IN},
                {"strokes": Output(PortType.COLOR), "aux": Output(PortType.COLOR)},
                {k: v for k, v in P.items() if v.group == ST or k.split("_")[-1] in
                 ("background", "shadows", "midtones", "highlights")})
FINAL = stage("paint_material_fx_oil_paint_final", "oil_paint.wgsl",
              {**material_inputs(CHANNELS), "src": COLOR_IN, "strokes": COLOR_IN, "aux": COLOR_IN}, OilPaint.outputs,
              {k: P[k] for k in ("effect_intensity", "fine_details", "sharpness", "roughness", "roughness_variation",
                                 "roughness_details", "normal_intensity", "normal_strokes_direction_influence",
                                 "amount", "thickness", "texture_intensity", "fibers_count")})
