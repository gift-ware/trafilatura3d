from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import mesh_inputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import B, F, I, MatFX, stage

CHANNELS = ("basecolor", "roughness", "metallic", "normal", "height", "ambient_occlusion")
BLEND = "Blending"
MR = "Blending/Metallic Roughness"


class RustWeathering(MatFX):
    """Rust over the material: it starts in the occluded areas and on the edges (baked ambient occlusion and
    curvature) and where the varnish is damaged (cells of Varnish Damage Scale), spreads (Rust Spreading, soft
    with Spreading Smoothness), is blurred (Blur Intensity) and drips down the model (world up = the position
    map's Y; image up without it): Drips Samples Amount steps over Drips Intensity, streaks as soft as Drips
    Smoothness. The rust (orange to dark brown, rough, not metallic, raised, crusty) is blended into each channel
    with its Blending intensity."""
    type = "paint_material_fx_rust_weathering"
    label = "Material FX Rust Weathering"
    catalog = "painter.filter.matfx_rust_weathering"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-rust-weathering"
    inputs = {**material_inputs(CHANNELS), **mesh_inputs("baked_ambient_occlusion", "curvature", "position")}
    outputs = material_outputs(CHANNELS)
    params = {"blur_intensity": F("Blur Intensity", 1.0, 0.0, 16.0), "blur_wrap": B("Blur Wrap", True),
              "rust_spreading": F("Rust Spreading", 0.4), "spreading_smoothness": F("Spreading Smoothness", 0.4),
              "varnish_damage_scale": F("Varnish Damage Scale", 0.5), "drips_intensity": F("Drips Intensity", 0.4),
              "drips_samples_amount": I("Drips Samples Amount", 16, 0, 32), "drips_smoothness": F("Drips Smoothness", 0.4),
              "normal_intensity": F("Normal Intensity", 1.0, group=BLEND),
              "ambient_occlusion_intensity": F("Ambient Occlusion Intensity", 1.0, group=BLEND),
              "height_intensity": F("Height Intensity", 1.0, group=BLEND),
              "base_color_intensity": F("Base Color Intensity", 1.0, group=MR),
              "roughness_intensity": F("Roughness Intensity", 1.0, group=MR),
              "metallic_intensity": F("Metallic Intensity", 1.0, group=MR)}
    unsupported = {}

    def encode(self, ctx, job):
        m = ctx.scratch(job, True)
        ctx.dispatch(PLACE, job, job.values, dsts={"rust": m})
        b = soft_blur(ctx, job, m, True, job.values["blur_intensity"], job.values["blur_wrap"])
        ctx.dispatch(FINAL, job, job.values, srcs={"rust": (b, True)})


P = RustWeathering.params
PLACE = stage("paint_material_fx_rust_weathering_place", "rust_weathering_place.wgsl",
              mesh_inputs("baked_ambient_occlusion", "curvature"), {"rust": Output(PortType.GRAY)},
              {k: P[k] for k in ("rust_spreading", "spreading_smoothness", "varnish_damage_scale")})
FINAL = stage("paint_material_fx_rust_weathering_final", "rust_weathering.wgsl",
              {**material_inputs(CHANNELS), **mesh_inputs("position"), "rust": Input(PortType.GRAY)},
              RustWeathering.outputs, {k: v for k, v in P.items() if k not in
                                       ("blur_intensity", "blur_wrap", "rust_spreading", "spreading_smoothness",
                                        "varnish_damage_scale")})
