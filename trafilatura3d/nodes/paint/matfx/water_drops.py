from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import mesh_inputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import F, MatFX, stage

CHANNELS = ("basecolor", "roughness", "metallic", "normal", "height")
WS = "World Space"
MAT = "Material"


class WaterDrops(MatFX):
    """Water drops scattered on the material (two sizes of cells, Drops Amount of them present, Drops Scale X / Y
    and Drops Scale Random), pulled by gravity: the world direction (Influence X, Y, Z of the position map; image
    down without it) orients the drops (Drops Rotation Vector Map Multiplier) and stretches a runoff trail above
    them (Drops Direction Intensity). Drops are domes: they refract the base color (Drops Base Color Warp
    Intensity), raise the height, bend the normal and take Drops Roughness / Drops Metallic (with their
    Blending). Water accumulates where the baked ambient occlusion is dark (AO Based Intensity, Spread, Intensity),
    wetting the base color and roughness. World Space masking weighs the drops by the world space normal's
    facing (Top, Bottom, Front, Back, Right, Left)."""
    type = "paint_material_fx_water_drops"
    label = "Material FX Water Drops"
    catalog = "painter.filter.matfx_water_drops"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-water-drops"
    inputs = {**material_inputs(CHANNELS), **mesh_inputs("baked_ambient_occlusion", "world_space_normals", "position")}
    outputs = material_outputs(CHANNELS)
    params = {
        "drops_amount": F("Drops Amount", 0.5), "drops_scale_x": F("Drops Scale X", 1.0, 0.1, 3.0),
        "drops_scale_y": F("Drops Scale Y", 1.0, 0.1, 3.0), "drops_scale_random": F("Drops Scale Random", 0.5),
        "drops_direction_intensity": F("Drops Direction Intensity", 0.3),
        "influence_x": F("Influence X", 0.0, -1.0, 1.0, "Position"),
        "influence_y": F("Influence Y", 1.0, -1.0, 1.0, "Position"),
        "influence_z": F("Influence Z", 0.0, -1.0, 1.0, "Position"),
        "intensity": F("Intensity", 0.3, group="Water Accumulation"),
        "spread": F("Spread", 2.0, 0.0, 16.0, "Water Accumulation"),
        "ao_based_intensity": F("AO Based Intensity", 0.5, group="Water Accumulation"),
        "masking_intensity": F("Masking Intensity", 0.5, group=WS), "top_intensity": F("Top Intensity", 1.0, group=WS),
        "bottom_intensity": F("Bottom Intensity", 0.2, group=WS), "front_intensity": F("Front Intensity", 1.0, group=WS),
        "back_intensity": F("Back Intensity", 1.0, group=WS), "right_intensity": F("Right Intensity", 1.0, group=WS),
        "left_intensity": F("Left Intensity", 1.0, group=WS),
        "drops_base_color_warp_intensity": F("Drops Base Color Warp Intensity", 0.5, group=MAT),
        "drops_rotation_vector_map_multiplier": F("Drops Rotation Vector Map Multiplier", 1.0, group=MAT),
        "drops_height_intensity": F("Drops Height Intensity", 0.5, group=MAT),
        "drops_roughness": F("Drops Roughness", 0.05, group=MAT),
        "drops_roughness_blending": F("Drops Roughness Blending", 1.0, group=MAT),
        "drops_metallic": F("Drops Metallic", 0.0, group=MAT),
        "drops_metallic_blending": F("Drops Metallic Blending", 0.0, group=MAT),
        "normal_intensity": F("Normal Intensity", 1.0, 0.0, 4.0, MAT),
    }
    unsupported = {}

    def encode(self, ctx, job):
        a = ctx.scratch(job, True)
        ctx.dispatch(ACC, job, job.values, dsts={"water": a})
        b = soft_blur(ctx, job, a, True, job.values["spread"])
        ctx.dispatch(FINAL, job, job.values, srcs={"water": (b, True)})


P = WaterDrops.params
ACC = stage("paint_material_fx_water_drops_acc", "water_drops_acc.wgsl", mesh_inputs("baked_ambient_occlusion"),
            {"water": Output(PortType.GRAY)}, {"ao_based_intensity": P["ao_based_intensity"]})
FINAL = stage("paint_material_fx_water_drops_final", "water_drops.wgsl",
              {**material_inputs(CHANNELS), **mesh_inputs("world_space_normals", "position"),
               "water": Input(PortType.GRAY)}, WaterDrops.outputs,
              {k: v for k, v in P.items() if k not in ("spread", "ao_based_intensity")})
