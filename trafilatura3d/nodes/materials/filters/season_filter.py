from trafilatura3d.nodes.materials._base import C, E, F, I, gray_in, material_inputs
from trafilatura3d.nodes.materials.filters._effect import EffectNode, advanced, blending

LABELS = ("Diffuse", "Base Color", "Normal", "Specular", "Glossiness", "Roughness", "Ambient Occlusion", "Height")


class SeasonFilter(EffectNode):
    """Legacy seasons effect (not PBR-correct): water filling the low areas of the Height (or the
    flat areas of the Normal) with fake refraction, reflection and ripples, freezing into cracked
    ice, snow on high flat areas, moss in the low ones, and a faked directional light on the color.
    Each channel takes the effect at its Blending intensity; Mask restricts it to the Mask input."""

    type = "season_filter"
    label = "Season Filter"
    category = "Material filters/Effects (Material)"
    catalog = "season_filter"
    inputs = {**material_inputs(), "mask": gray_in("Mask", 1.0)}
    params = {
        **advanced(),
        "light_intensity": F(0.5, "Light Intensity", 0.0, 1.0, group="Advanced"),
        "light_angle": F(0.125, "Light Angle", 0.0, 1.0, group="Advanced"),
        "effect_from_height_or_normal": E("Height", "Effect From Height or Normal", ("Height", "Normal"), group="Effect"),
        "water_level": F(0.0, "Water Level", 0.0, 1.0, group="Effect"),
        "water_details": F(0.0, "Water Details", 0.0, 1.0, group="Effect"),
        "refraction": F(0.5, "Refraction", 0.0, 1.0, group="Effect"),
        "reflection": F(0.5, "Reflection", 0.0, 1.0, group="Effect"),
        "reflection_distance": F(0.5, "Reflection Distance", 0.0, 1.0, group="Effect"),
        "reflection_angle": F(0.625, "Reflection Angle", 0.0, 1.0, group="Effect"),
        "flow_direction": F(0.0, "Flow Direction", 0.0, 1.0, group="Effect"),
        "ice": F(0.47, "Ice", 0.0, 1.0, group="Effect"),
        "ice_details": F(1.0, "Ice Details", 0.0, 1.0, group="Effect"),
        "snow": F(0.0, "Snow", 0.0, 1.0, group="Effect"),
        "moss": F(0.0, "Moss", 0.0, 1.0, group="Effect"),
        "moss_scale": I(3, "Moss Scale", 1, 4, group="Effect"),
        "moss_color": C([0.18, 0.3, 0.06], "Moss Color", group="Effect"),
        "water_color": C([0.1, 0.2, 0.22, 0.6], "Water Color", group="Effect"),
        **blending(LABELS, defaults={"Ambient Occlusion": 0.3}),
    }
