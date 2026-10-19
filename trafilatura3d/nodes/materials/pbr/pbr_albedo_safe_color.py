from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, E, F, MatNode, color_in, gray_in
from trafilatura3d.nodes.materials.pbr._pbr import CATEGORY


class PbrAlbedoSafeColor(MatNode):
    """Brings Base Color (or Diffuse) values into the PBR-safe range by scaling each pixel's color
    (hue kept): dielectrics between 50 and 240 sRGB of luminance, metals (by the Metallic input, in
    the Base Color - Metallic workflow) at least 70% reflective. Tolerance widens the range (down
    to 30 sRGB, up to 255, metals to 50%)."""

    type = "pbr_albedo_safe_color"
    label = "PBR Albedo Safe Color"
    category = CATEGORY
    catalog = "pbr_albedo_safe_color"
    inputs = {"basecolor": color_in("Base Color / Diffuse", (0.5, 0.5, 0.5, 1.0)), "metallic": gray_in("Metallic", 0.0)}
    outputs = {"basecolor": Output(COLOR, "Base Color / Diffuse")}
    params = {
        "pbr_workflow": E("Base Color - Metallic", "PBR Workflow", ("Base Color - Metallic", "Diffuse - Specular")),
        "tolerance": F(0.5, "Tolerance", 0.0, 1.0),
    }
