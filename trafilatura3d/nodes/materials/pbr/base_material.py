from trafilatura3d.nodes.materials._base import (CHANNELS, B, C, E, F, I, MatNode, gray_in, material_inputs,
                                           material_outputs)
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY, DIELECTRICS, METALS

USER_MAPS = ("basecolor", "normal", "roughness", "metallic", "height", "ambient_occlusion", "diffuse", "specular",
             "glossiness", "emissive", "opacity")
# The User-defined maps switches: (our key, channel), in order.
USER = (("user_diffuse", "diffuse"), ("user_basecolor", "basecolor"), ("user_normal", "normal"),
        ("user_specular", "specular"), ("user_glossiness", "glossiness"), ("user_roughness", "roughness"),
        ("user_metallic", "metallic"), ("user_ao", "ambient_occlusion"), ("user_height", "height"))


class BaseMaterial(MatNode):
    """A full material from solid values: PBR workflow Metal-Roughness (Base Color, Metallic,
    Roughness) or Specular-Glossiness (Diffuse, Specular, Glossiness), every channel of both written
    consistently. Material Preset sets a pure metal (or a dielectric whose specular comes from
    Specular Presets / Range); a grunge (built-in or custom, tiled by Grunge Tiling) varies roughness
    and glossiness by their own Grunge Amount. A user-defined map replaces the solid value when its
    switch is on (emissive and opacity, ours, when connected); a custom height is placed by Height
    Position / Range and converted to the normal by Normal from Height Intensity."""

    type = "base_material"
    main_input = "normal"  # the library graph's primary input: Relative to input follows it
    label = "Base Material"
    category = CATEGORY
    catalog = "base_material"
    includes = ("_pbr.wgsl",)
    inputs = {**material_inputs(USER_MAPS), "grunge": gray_in("Grunge", 0.5)}
    outputs = material_outputs()
    params = {
        "pbr_workflow": E("Metal - Roughness", "PBR Workflow", ("Metal - Roughness", "Specular - Glossiness")),
        "material_preset": E("Custom", "Material Preset", ("Custom", "Dielectric") + METALS),
        "base_color": C([0.498, 0.498, 0.498], "Base Color"),
        "metallic": F(1.0, "Metallic", 0.0, 1.0),
        "diffuse_color": C([0.498, 0.498, 0.498], "Diffuse Color"),
        "specular": C([0.498, 0.498, 0.498], "Specular"),
        "specular_presets": E("Plastic", "Specular Presets", DIELECTRICS),
        "specular_range": F(0.5, "Specular Range", 0.0, 1.0),
        "roughness_value": F(0.251, "Roughness Value", 0.0, 1.0, group="Roughness - Glossiness"),
        "grunge_amount": F(0.0, "Grunge Amount", 0.0, 1.0, group="Roughness - Glossiness"),
        "glossiness_value": F(0.753, "Glossiness Value", 0.0, 1.0, group="Roughness - Glossiness"),
        "glossiness_grunge_amount": F(0.0, "Grunge Amount", 0.0, 1.0, group="Roughness - Glossiness"),
        "grunge_tiling": I(1, "Grunge Tiling", 1, 16, group="Roughness - Glossiness"),
        "custom_grunge_input": B(False, "Custom Grunge Input", group="Roughness - Glossiness"),
        "normal_from_height_intensity": F(1.0, "Normal from Height Intensity", 0.0, 16.0, group="Normal"),
        "height_position": F(0.5, "Height Position", 0.0, 1.0, group="Height"),
        "height_range": F(0.5, "Height Range", 0.0, 1.0, group="Height"),
        **{k: B(False, CHANNELS[ch][0], group="User-defined Maps") for k, ch in USER},
    }
