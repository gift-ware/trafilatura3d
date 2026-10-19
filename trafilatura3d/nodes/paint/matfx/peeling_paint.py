from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import mesh_inputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import B, F, MatFX, stage

CHANNELS = ("basecolor", "roughness", "normal", "height", "ambient_occlusion")
TECH = "Technical parameters"


class PeelingPaint(MatFX):
    """Paint peeling off: a peel field (noise, flake cracks of Flaking Density stretched by Flaking X / Y Amount,
    smaller flakes with Flake Level, the edges of the curvature map reached within Curvature Distance with Use
    Curvature) is blurred (Blur Intensity) and cut at Peeling Level. Peeled areas lose their coverage (alpha 0 in
    every channel): inside a group the layers below the group show through (documentation). Around the holes the
    paint curls up over Peeling Distance (height, normal, darker ambient occlusion under it); air bubbles raise
    small blisters. Luminosity, Contrast, Hue Shift and Saturation change the paint's top surface."""
    type = "paint_material_fx_peeling_paint"
    label = "Material FX Peeling Paint"
    catalog = "painter.filter.matfx_peeling_paint"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-peeling-paint"
    inputs = {**material_inputs(CHANNELS), **mesh_inputs("curvature")}
    outputs = material_outputs(CHANNELS)
    params = {
        "blur_intensity": F("Blur Intensity", 0.5, 0.0, 16.0), "blur_wrap": B("Blur Wrap", True),
        "peeling_level": F("Peeling Level", 0.35), "peeling_distance": F("Peeling Distance", 0.3),
        "flake_level": F("Flake Level", 0.3), "air_bubble_density": F("Air Bubble Density", 0.3),
        "flaking_density": F("Flaking Density", 0.5), "flaking_x_amount": F("Flaking X Amount", 0.5),
        "flaking_y_amount": F("Flaking Y Amount", 0.5), "use_curvature": B("Use Curvature", True),
        "curvature_distance": F("Curvature Distance", 2.0, 0.0, 16.0), "curvature_contrast": F("Curvature Contrast", 0.5),
        "luminosity": F("Luminosity", 0.5, group=TECH), "contrast": F("Contrast", 0.0, -1.0, 1.0, TECH),
        "hue_shift": F("Hue Shift", 0.0, -0.5, 0.5, TECH), "saturation": F("Saturation", 0.5, group=TECH),
        "normal_intensity": F("Normal Intensity", 1.0, 0.0, 4.0, TECH),
        "height_range": F("Height Range", 0.5, group=TECH), "height_position": F("Height Position", 0.5, group=TECH),
        "ambient_occlusion_intensity": F("Ambient Occlusion Intensity", 0.5, group=TECH),
    }
    unsupported = {}

    def encode(self, ctx, job):
        f = ctx.scratch(job, True)
        ctx.dispatch(FIELD, job, job.values, dsts={"peel": f})
        b = soft_blur(ctx, job, f, True, job.values["blur_intensity"], job.values["blur_wrap"])
        ctx.dispatch(FINAL, job, job.values, srcs={"peel": (b, True)})


P = PeelingPaint.params
FIELD = stage("paint_material_fx_peeling_paint_field", "peeling_paint_field.wgsl", mesh_inputs("curvature"),
              {"peel": Output(PortType.GRAY)},
              {k: P[k] for k in ("flake_level", "flaking_density", "flaking_x_amount", "flaking_y_amount",
                                 "use_curvature", "curvature_distance", "curvature_contrast")})
FINAL = stage("paint_material_fx_peeling_paint_final", "peeling_paint.wgsl",
              {**material_inputs(CHANNELS), "peel": Input(PortType.GRAY)}, PeelingPaint.outputs,
              {k: P[k] for k in ("peeling_level", "peeling_distance", "air_bubble_density", "luminosity", "contrast",
                                 "hue_shift", "saturation", "normal_intensity", "height_range", "height_position",
                                 "ambient_occlusion_intensity")})
