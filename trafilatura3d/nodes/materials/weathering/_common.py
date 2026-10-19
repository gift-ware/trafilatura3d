from trafilatura3d.nodes.materials._base import color_in, gray_in, material_inputs
from trafilatura3d.nodes.materials.filters._effect import CHANNEL_BLENDING, EffectNode, advanced, blending

CATEGORY = "Mesh based generators/Weathering"
NO_METALLIC = tuple(l for l in CHANNEL_BLENDING if l != "Metallic")
BAKED_AO = gray_in("Ambient Occlusion", 1.0)
BAKED_CURV = gray_in("Curvature", 0.5)
BAKED_WSN = color_in("Normal World Space", (0.5, 1.0, 0.5, 1.0))
BAKED_POS = color_in("Position", (0.5, 0.5, 0.5, 1.0))
BAKED_HEIGHT = gray_in("Height", 0.5)
MASK = gray_in("Mask", 1.0)


class WeatheringNode(EffectNode):
    category = CATEGORY
    includes = ("../_material.wgsl", "_weather.wgsl")
