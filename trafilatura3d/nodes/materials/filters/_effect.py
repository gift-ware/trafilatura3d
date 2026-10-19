from trafilatura3d.nodes.materials._base import (NORMAL_FORMATS, B, E, F, MatNode, channel_toggles, gray_in,
                                           material_inputs, material_outputs)

CHANNEL_BLENDING = ("Diffuse", "Base Color", "Normal", "Specular", "Glossiness", "Roughness", "Metallic",
                    "Ambient Occlusion", "Height")
# The channels a material effect (weathering, season) processes, each with its switch (on by default).
EFFECT_CHANNELS = ("diffuse", "basecolor", "normal", "specular", "glossiness", "roughness", "metallic", "height",
                   "ambient_occlusion")
# The surface channels switched in Snow Cover and Water Level, the metal-roughness ones on by default.
SURFACE_CHANNELS = ("basecolor", "diffuse", "roughness", "glossiness", "metallic", "specular")
SURFACE_ON = ("basecolor", "roughness", "metallic")


def blending(labels=CHANNEL_BLENDING, normal_max=1.0, defaults=None) -> dict:
    """The Blending group: "<Channel> Intensity" per channel, 1 unless `defaults` ({label: value})."""
    out = {}
    for lab in labels:
        name = lab.lower().replace(" ", "_") + "_intensity"
        hi = normal_max if lab == "Normal" else 1.0
        out[name] = F((defaults or {}).get(lab, 1.0), f"{lab} Intensity", 0.0, hi, group="Blending")
    return out


def advanced() -> dict:
    """The Channels switches and the Advanced group of a material effect."""
    return {**channel_toggles(EFFECT_CHANNELS, EFFECT_CHANNELS),
            "normal_format": E("OpenGL", "Normal Format", NORMAL_FORMATS, group="Advanced"),
            "mask": B(False, "Mask", group="Advanced")}


class EffectNode(MatNode):
    """A full-material effect: the material in, the material out (every channel), shader built on
    `_material.wgsl` (read_mat / blend_mat / write_mat)."""
    includes = ("../_material.wgsl",)
    outputs = material_outputs()
