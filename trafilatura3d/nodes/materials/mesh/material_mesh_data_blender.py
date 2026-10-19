from trafilatura3d.nodes.materials._base import (FULL, B, E, F, I, MatNode, Pass, channel_passes, channel_toggles, color_in,
                                           gray_in, material_inputs, material_outputs)
from trafilatura3d.nodes.materials.filters._effect import EFFECT_CHANNELS
from trafilatura3d.nodes.materials.mesh._common import CATEGORY

BAKED = {"baked_ambient_occlusion": gray_in("Ambient Occlusion", 1.0), "baked_curvature": gray_in("Curvature", 0.5),
         "baked_height": gray_in("Height", 0.5), "baked_normal": color_in("Normal", (0.5, 0.5, 1.0, 1.0)),
         "baked_vertex_color": color_in("Vertex Color", (1.0, 1.0, 1.0, 1.0)),
         "baked_world_space_normal": color_in("World Space Normal", (0.5, 1.0, 0.5, 1.0))}
G = "Baked Maps"


class MaterialMeshDataBlender(MatNode):
    """Adds baked-map detail to a full material: AO darkening diffuse / base color / specular, convex
    edges from the curvature lightening colors and specular, darkening glossiness, roughening and
    metallizing, vertex colors, fake pre-lighting from the world space normals (realistic or
    cartoon, banded in layers, with curvature outlines), the baked tangent normal under the
    material's, baked AO and height combined with the material's. Baked maps are the `baked_` ports, each with
    its switch (off: its neutral value); a channel switched off passes through."""

    type = "material_mesh_data_blender"
    label = "Material Mesh Data Blender"
    category = CATEGORY
    catalog = "material_mesh_data_blender"
    inputs = {**material_inputs(), **BAKED}
    outputs = material_outputs()
    params = {
        **channel_toggles(EFFECT_CHANNELS, EFFECT_CHANNELS),
        **{f"use_{k}": B(True, v.label, group=G) for k, v in BAKED.items()},
        "diffuse_ao": F(0.0, "Diffuse AO", 0.0, 1.0, group=G),
        "diffuse_sharp_edges": F(0.0, "Diffuse Sharp Edges", 0.0, 1.0, group=G),
        "diffuse_color_from_vertex_color": F(0.0, "Diffuse Color From Vertex Color", 0.0, 1.0, group=G),
        "diffuse_pre_lighting": F(0.0, "Diffuse Pre Lighting", 0.0, 1.0, group=G),
        "diffuse_cartoon_lighting_balance": F(1.0, "Diffuse Cartoon Lighting Balance", 0.0, 1.0, group=G),
        "diffuse_cartoon_pre_lighting_layers": I(4, "Diffuse Cartoon Pre Lighting Layers", 0, 10, group=G),
        "diffuse_cartoon_outlines": F(0.0, "Diffuse Cartoon Outlines", 0.0, 1.0, group=G),
        "base_color_ao": F(0.0, "Base Color AO", 0.0, 1.0, group=G),
        "base_color_sharp_edges": F(0.0, "Base Color Sharp edges", 0.0, 1.0, group=G),
        "base_color_from_vertex_color": F(0.0, "Base Color From Vertex Color", 0.0, 1.0, group=G),
        "normal_material_intensity": F(0.0, "Normal Material Intensity", 0.0, 1.0, group=G),
        "specularao": F(0.0, "SpecularAO", 0.0, 1.0, group=G),
        "specular_bright_sharp_edges": F(0.0, "Specular Bright Sharp Edges", 0.0, 1.0, group=G),
        "specular_cartoon_outlines": F(0.0, "Specular Cartoon Outlines", 0.0, 1.0, group=G),
        "glossiness_dark_sharp_edges": F(0.0, "Glossiness Dark Sharp Edges", 0.0, 1.0, group=G),
        "roughness_bright_sharp_edges": F(0.0, "Roughness Bright Sharp Edges", 0.0, 1.0, group=G),
        "roughness_cartoon_outlines": F(0.0, "Roughness Cartoon Outlines", 0.0, 1.0, group=G),
        "metallic_bright_sharp_edges": F(0.0, "Metallic Bright Sharp Edges", 0.0, 1.0, group=G),
        "metallic_cartoon_outlines": F(0.0, "Metallic Cartoon Outlines", 0.0, 1.0, group=G),
        "ao_materiel_intensity": F(0.0, "AO Materiel Intensity", 0.0, 1.0, group=G),
        "height_material_intensity": F(0.0, "Height Material Intensity", 0.0, 1.0, group=G),
        "height_material_blending_type": E("Reinforce", "Height Material Blending Type", ("Reinforce", "Interpolation"),
                                           group=G),
    }

    def encode(self, ctx, job):
        ps = channel_passes(self, lambda n, ch: Pass(
            n, ch, "material_mesh_data_blender.wgsl", {ch: n.inputs[ch], **BAKED}, {ch: n.outputs[ch]}, channel=ch,
            subst={"$CHID": str(FULL.index(ch))}))
        for p in ps.values():
            ctx.dispatch(p, job, job.values)
