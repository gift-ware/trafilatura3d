from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import C, E, F, mesh_inputs
from trafilatura3d.nodes.paint.matfinish._lighting import LIGHTING_DOC, MESH, Lighting, grouped


def light(n: int, h: float, v: float) -> dict:
    return grouped(f"Light {n}", {
        f"light{n}_horizontal_angle": F("Horizontal Angle", h, 0.0, 360.0),
        f"light{n}_vertical_angle": F("Vertical Angle", v, -90.0, 90.0),
        f"light{n}_intensity": F("Intensity", 0.0, 0.0, 4.0),
        f"light{n}_color": C("Color", (1.0, 1.0, 1.0))})


class BakedLightingStylized(Lighting):
    """The material lit by a sun, a sky / horizon / ground gradient and two more lights, baked into the color:
    ambient occlusion and cavities (concave curvature) darken the diffuse and the specular lighting, convex edges
    (curvature) are brightened. Output Lighting Only: the lighting of a white material, without the color."""
    type = "paint_baked_lighting_stylized"
    label = "Baked Lighting Stylized"
    catalog = "painter.filter.baked_lighting_stylized"
    doc = LIGHTING_DOC + "baked-lighting-stylized"
    inputs = {**material_inputs(("basecolor", "roughness", "metallic", "diffuse", "specular", "glossiness",
                                 "normal", "height")),
              **mesh_inputs(*MESH, "curvature")}
    outputs = material_outputs(("basecolor", "diffuse"))
    params = {
        "input": E("Input", ("Metallic Roughness", "Specular Glossiness")),
        "output": E("Output", ("Lit Color", "Lighting Only")),
        **grouped("Material", {
            "dielectric_reflectance": F("Dielectric Reflectance", 0.04, 0.0, 0.2),
            "diffuse_ao": F("Diffuse AO", 1.0, 0.0, 1.0), "diffuse_cavity": F("Diffuse Cavity", 0.5, 0.0, 1.0),
            "specular_ao": F("Specular AO", 1.0, 0.0, 1.0), "specular_cavity": F("Specular Cavity", 0.5, 0.0, 1.0),
            "cavity_smoothness": F("Cavity Smoothness", 0.5, 0.0, 1.0),
            "edges_intensity": F("Edges Intensity", 0.3, 0.0, 1.0),
            "edges_smoothness": F("Edges Smoothness", 0.5, 0.0, 1.0),
            "normal_details_type": E("Normal Details Type", ("Mesh Only", "Mesh + Height + Normal"),
                                     "Mesh + Height + Normal"),
            "height_to_normal_intensity": F("Height to Normal Intensity", 0.5, 0.0, 1.0)}),
        **grouped("Sun & Sky", {
            "sun_intensity": F("Sun Intensity", 1.0, 0.0, 4.0),
            "sun_horizontal_angle": F("Sun Horizontal Angle", 45.0, 0.0, 360.0),
            "sun_vertical_angle": F("Sun Vertical Angle", 45.0, -90.0, 90.0),
            "sun_color": C("Sun Color", (1.0, 0.95, 0.85)),
            "sky_intensity": F("Sky Intensity", 0.5, 0.0, 4.0),
            "sky_color": C("Sky Color", (0.55, 0.7, 1.0)),
            "horizon_color": C("Horizon Color", (0.85, 0.85, 0.85)),
            "ground_color": C("Ground Color", (0.35, 0.3, 0.25))}),
        **light(1, 225.0, 20.0),
        **light(2, 135.0, -20.0),
    }
