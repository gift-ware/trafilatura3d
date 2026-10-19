from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.filters.basic._lib import gaussian
from trafilatura3d.nodes.materials._base import (GRAY, NORMAL_FORMATS, B, E, F, F2, F3, F4, MatNode, Pass, channel_toggles,
                                           color_in, gray_in, material_outputs, mean, passes_of, px256)

# The Channels group (no Base Color switch: its output is the adjusted diffuse).
CHANNELS = ("diffuse", "normal", "specular", "roughness", "glossiness", "metallic", "ambient_occlusion", "height")


class BitmapToMaterialLight(MatNode):
    """A full material from one Diffuse/Basecolor image. Relief: the luminance split in four frequency
    bands (large scale to fine detail) weighted by the Relief Equalizer (x = largest), Pinch sharpens,
    shifted by Depth Balance, as Height and Normal (Normal Strength, Sharpen, Soften). Ambient occlusion
    from how far each pixel lies under its blurred surroundings (Spread, Light Distance), offset towards
    Light Angle, through its output Levels. Specular, Glossiness and Roughness from the luminance
    through Levels (In: low, mid, high; Out: low, high), the optional Metallic input pulling them
    towards metal values. Diffuse adjusted (hue, saturation, brightness, contrast, sharpen) and
    darkened by the AO. A channel switched off gives its default value (the Metallic input for
    Metallic)."""

    type = "image_to_material_light"
    label = "Image to Material Light"
    category = "Material filters/1-Click"
    catalog = "bitmap_to_material_light"
    inputs = {"basecolor": color_in("Diffuse / Base Color", (0.5, 0.5, 0.5, 1.0)), "metallic": gray_in("Metallic", 0.0)}
    outputs = material_outputs()
    params = {
        **channel_toggles(CHANNELS, CHANNELS),
        "depth_balance": F(0.0, "Depth Balance", -1.0, 1.0, group="Global"),
        "sharpen": F(0.0, "Sharpen", 0.0, 1.0, group="Diffuse"),
        "hue": F(0.0, "Hue", 0.0, 1.0, group="Diffuse"),
        "saturation": F(0.5, "Saturation", 0.0, 1.0, group="Diffuse"),
        "brightness": F(0.5, "Brightness", 0.0, 1.0, group="Diffuse"),
        "contrast": F(0.0, "Contrast", -1.0, 1.0, group="Diffuse"),
        "output_normal_format": E("OpenGL", "Output Normal Format", NORMAL_FORMATS, group="Relief"),
        "invert_generated_relief": B(False, "Invert Generated Relief", group="Relief"),
        "normal_strength": F(10.0, "Normal Strength", 0.0, 20.0, group="Relief"),
        "relief_equalizer": F4([0.03, 0.25, 0.54, 1.0], "Relief Equalizer", 0.0, 1.0, group="Relief"),
        "pinch_intensity": F(0.0, "Pinch Intensity", 0.0, 1.0, group="Relief"),
        "normal_sharpen": F(0.0, "Normal Sharpen", 0.0, 1.0, group="Relief"),
        "normal_soften": F(0.0, "Normal Soften", 0.0, 1.0, group="Relief"),
        "specular_diffuse_influence": F(0.0, "Specular Diffuse Influence", 0.0, 1.0, group="Specular"),
        "specular_saturation": F(0.5, "Specular Saturation", 0.0, 1.0, group="Specular"),
        "specular_sharpen": F(0.0, "Specular Sharpen", 0.0, 1.0, group="Specular"),
        "specular_levels_in": F3([0.3, 0.6, 0.81], "Specular Levels In", 0.0, 1.0, group="Specular"),
        "specular_levels_out": F2([0.0, 1.0], "Specular Levels Out", 0.0, 1.0, group="Specular"),
        "metallic_specular_influence": F(0.0, "Metallic Specular Influence", 0.0, 1.0, group="Specular"),
        "glossiness_levels_in": F3([0.21, 0.5, 0.82], "Glossiness Levels In", 0.0, 1.0, group="Glossiness"),
        "glossiness_levels_out": F2([0.08, 0.68], "Glossiness Levels Out", 0.0, 1.0, group="Glossiness"),
        "metallic_glossiness_influence": F(0.0, "Metallic Glossiness Influence", 0.0, 1.0, group="Glossiness"),
        "roughness_levels_in": F3([0.4, 0.5, 0.6], "Roughness Levels In", 0.0, 1.0, group="Roughness"),
        "roughness_levels_out": F2([0.0, 1.0], "Roughness Levels Out", 0.0, 1.0, group="Roughness"),
        "metallic_roughness_influence": F(0.0, "Metallic Roughness Influence", 0.0, 1.0, group="Roughness"),
        "ambient_occlusion_in_diffuse": F(0.0, "Ambient Occlusion In Diffuse", 0.0, 1.0, group="Ambient Occlusion"),
        "ambient_occlusion_spread": F(1.0, "Ambient Occlusion Spread", 0.0, 1.0, group="Ambient Occlusion"),
        "ambient_occlusion_light_distance": F(0.0, "Ambient Occlusion Light Distance", 0.0, 1.0,
                                              group="Ambient Occlusion"),
        "ambient_occlusion_light_angle": F(0.13, "Ambient Occlusion Light Angle", 0.0, 1.0, group="Ambient Occlusion"),
        "ambient_occlusion_levels": F2([0.0, 1.0], "Ambient Occlusion Levels", 0.0, 1.0, group="Ambient Occlusion"),
    }

    def encode(self, ctx, job):
        g = Input(GRAY)
        lum, height, final = passes_of(self, lambda n: (
            Pass(n, "lum", "image_to_material_light_lum.wgsl", {"basecolor": n.inputs["basecolor"]},
                 {"l": Output(GRAY)}),
            Pass(n, "height", "image_to_material_light_height.wgsl", {"l": g, "b1": g, "b2": g, "b3": g, "lm": g},
                 {"h": Output(GRAY)}),
            Pass(n, "final", "image_to_material_light.wgsl",
                 {**n.inputs, "h": g, "hb": g, "hs": g, "lb": g}, n.outputs)))
        k = px256(job)
        L = ctx.scratch(job, True)
        ctx.dispatch(lum, job, job.values, dsts={"l": L})
        b1 = gaussian(ctx, job, L, True, 2.0 * k)
        b2 = gaussian(ctx, job, L, True, 24.0 * k)
        b3 = gaussian(ctx, job, L, True, 6.0 * k)
        H = ctx.scratch(job, True)
        ctx.dispatch(height, job, job.values, srcs={"l": (L, True), "b1": (b1, True), "b2": (b2, True),
                                                    "b3": (b3, True), "lm": (mean(ctx, job, L, True), True)},
                     dsts={"h": H})
        hb = gaussian(ctx, job, H, True, (job.values["ambient_occlusion_spread"] * 24.0 + 2.0) * k)
        hs = gaussian(ctx, job, H, True, 1.5 * k)
        ctx.dispatch(final, job, job.values, srcs={"h": (H, True), "hb": (hb, True), "hs": (hs, True), "lb": (b1, True)})
