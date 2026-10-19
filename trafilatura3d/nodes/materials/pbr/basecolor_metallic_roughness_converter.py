from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, GRAY, B, E, MatNode, color_in, gray_in
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY

TARGETS = ("PBR Diffuse/Specular/Gloss", "Vray (GGX)", "Corona", "Corona 1.6+", "Redshift 1.x",
           "Arnold 4 (AiStandard)", "Arnold 4 (AlSurface)", "RenderMan (PxrSurface)")


class BasecolorMetallicRoughnessConverter(MatNode):
    """Converts Base Color / Metallic / Roughness to other models: Diffuse (albedo of the dielectric
    part), Specular (reflection color: F0 blended to the base color for metals, or white with
    Fresnel-by-IOR renderers: Vray, Corona, Redshift), Glossiness and Roughness in the target's
    convention (Corona < 1.6 and Arnold 4 AiStandard remap them). F0 of dielectrics: 0.04, or
    0.08 x Specular Level when the SpecularLevel input is used. Approximations of each renderer's
    documented parameters."""

    type = "basecolor_metallic_roughness_converter"
    label = "BaseColor / Metallic / Roughness converter"
    category = CATEGORY
    catalog = "basecolor_metallic_roughness_converter"
    inputs = {"basecolor": color_in("Base Color", (0.5, 0.5, 0.5, 1.0)), "metallic": gray_in("Metallic", 0.0),
              "roughness": gray_in("Roughness", 0.5), "specular_level": gray_in("Specular Level", 0.5)}
    outputs = {"diffuse": Output(COLOR, "Diffuse"), "specular": Output(COLOR, "Specular"),
               "glossiness": Output(GRAY, "Glossiness"), "roughness": Output(GRAY, "Roughness")}
    params = {
        "use_specularlevel_input": B(False, "Use SpecularLevel input"),
        "target_model": E("PBR Diffuse/Specular/Gloss", "Target", TARGETS),
    }
