from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import B, C, E, F, MatFX, stage

READ = ("basecolor", "metallic", "roughness", "diffuse", "specular", "glossiness", "normal", "height")
WRITE = ("basecolor", "metallic", "roughness", "diffuse", "specular", "glossiness", "normal")
PBR_MR = "Material/PBR Metallic Roughness"
PBR_SG = "Material/PBR Specular Glossiness"


class DetailEdgeWear(MatFX):
    """Worn edges found in the stack below: curvature of the normal channel (Input Mode Normal; flat normals give
    no wear, as documented) or of the height channel, scaled by Curvature Intensity, convex edges (Invert Curvature:
    the cavities, Combine Curvature: both), smoothed by Blur Intensity. The wear (Wear Level, Wear Contrast, Edges
    Smoothness) is broken by a grunge (Grunge Amount, Grunge Scale) and kept out of the cavities spread by AO
    Spreading; the resulting generator mask is blurred (Generator Mask Blur) then leveled (Generator Mask Control,
    Contrast); outside the stack's coverage it takes Alpha Background Value. Worn areas show the material of the
    Material group (bare metal by default) and a small step in the normal (Normal Intensity)."""
    type = "paint_material_fx_detail_edge_wear"
    label = "Material FX Detail Edge Wear"
    catalog = "painter.filter.matfx_detail_edge_wear"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-detail-edge-wear"
    inputs = material_inputs(READ)
    outputs = material_outputs(WRITE)
    params = {
        "blur_intensity": F("Blur Intensity", 1.0, 0.0, 16.0), "blur_wrap": B("Blur Wrap", True),
        "input_mode": E("Input Mode", ("Normal", "Height")),
        "wear_level": F("Wear Level", 0.5), "wear_contrast": F("Wear Contrast", 0.5),
        "edges_smoothness": F("Edges Smoothness", 0.3), "grunge_amount": F("Grunge Amount", 0.5),
        "grunge_scale": F("Grunge Scale", 1.0, 0.1, 8.0),
        "base_color": C("Base Color", (0.78, 0.77, 0.75), PBR_MR), "metallic": F("Metallic", 1.0, group=PBR_MR),
        "roughness": F("Roughness", 0.3, group=PBR_MR),
        "diffuse": C("Diffuse", (0.0, 0.0, 0.0), PBR_SG), "specular_color": C("Specular Color", (0.78, 0.77, 0.75), PBR_SG),
        "glossiness": F("Glossiness", 0.7, group=PBR_SG),
        "generator_mask_control": F("Generator Mask Control", 0.5, group="Settings"),
        "generator_mask_contrast": F("Generator Mask Contrast", 0.5, group="Settings"),
        "generator_mask_blur": F("Generator Mask Blur", 0.0, 0.0, 16.0, group="Settings"),
        "alpha_background_value": F("Alpha Background Value", 0.0, group="Settings"),
        "curvature_intensity": F("Curvature Intensity", 1.0, 0.0, 4.0, group="Settings"),
        "invert_curvature": B("Invert Curvature", False, group="Settings"),
        "combine_curvature": B("Combine Curvature", False, group="Settings"),
        "normal_intensity": F("Normal Intensity", 0.5, group="Settings"),
        "ao_spreading": F("AO Spreading", 2.0, 0.0, 16.0, group="Settings"),
    }
    unsupported = {}

    def encode(self, ctx, job):
        v = job.values
        cv = ctx.scratch(job, False)
        ctx.dispatch(CURV, job, v, dsts={"curv": cv})
        edges = soft_blur(ctx, job, cv, False, v["blur_intensity"], v["blur_wrap"])
        occl = soft_blur(ctx, job, cv, False, v["ao_spreading"], v["blur_wrap"])
        m = ctx.scratch(job, True)
        ctx.dispatch(MASK, job, v, srcs={"edges": (edges, False), "occlusion": (occl, False)}, dsts={"mask": m})
        mb = soft_blur(ctx, job, m, True, v["generator_mask_blur"], v["blur_wrap"])
        ctx.dispatch(FINAL, job, v, srcs={"mask": (mb, True)})


P = DetailEdgeWear.params
CURV = stage("paint_material_fx_detail_edge_wear_curv", "detail_edge_wear_curv.wgsl", material_inputs(("normal", "height")),
             {"curv": Output(PortType.COLOR)},
             {k: P[k] for k in ("input_mode", "curvature_intensity", "invert_curvature", "combine_curvature")})
MASK = stage("paint_material_fx_detail_edge_wear_mask", "detail_edge_wear_mask.wgsl",
             {"edges": Input(PortType.COLOR), "occlusion": Input(PortType.COLOR), **material_inputs(("normal",))},
             {"mask": Output(PortType.GRAY)},
             {k: P[k] for k in ("wear_level", "wear_contrast", "edges_smoothness", "grunge_amount", "grunge_scale",
                                "alpha_background_value")})
FINAL = stage("paint_material_fx_detail_edge_wear_final", "detail_edge_wear.wgsl",
              {**material_inputs(WRITE), "mask": Input(PortType.GRAY)}, DetailEdgeWear.outputs,
              {k: P[k] for k in ("base_color", "metallic", "roughness", "diffuse", "specular_color", "glossiness",
                                 "generator_mask_control", "generator_mask_contrast", "normal_intensity")})
