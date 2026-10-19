from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, B, E, MatNode, color_in, gray_in
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY


class PbrBasecolorMetallicValidate(MatNode):
    """Good-to-bad heatmap of PBR correctness: green = correct, yellow = borderline, red = wrong.
    Albedo: dielectric base color luminance between the dark threshold (50 or 30 sRGB) and 240 sRGB.
    Metal: metal pixels must reflect at least 70% (or 60%) and metallic should be 0 or 1 (in-between
    values show orange). Combined shows the worse of both. Overlay Map multiplies the heatmap with the
    base color's luminance to locate problems."""

    type = "pbr_basecolor_metallic_validate"
    label = "PBR BaseColor / Metallic Validate"
    category = CATEGORY
    catalog = "pbr_basecolor_metallic_validate"
    inputs = {"basecolor": color_in("Base Color", (0.5, 0.5, 0.5, 1.0)), "metallic": gray_in("Metallic", 0.0)}
    outputs = {"output": Output(COLOR)}
    params = {
        "validation_mode": E("Combined", "Validation Mode", ("Albedo", "Metal", "Combined")),
        "albedo_dark_range_threshold": E("50 sRGB", "Albedo Dark Range Threshold", ("50 sRGB", "30 sRGB")),
        "metal_reflectance_range": E("70-100% Reflective", "Metal Reflectance Range",
                                     ("70-100% Reflective", "60-100% Reflective")),
        "overlay_map": B(False, "Overlay Map"),
    }
