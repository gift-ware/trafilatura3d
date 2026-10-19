"""Texture Set channels (the document channels): identifiers, labels, storage formats, defaults.

From the official documentation: the channel tags of the shader API (all-engine-params), the storage
formats of Texture Set settings (sRGB8 … RGB32F) and the shader defaults of lib-sampler (DEFAULT_*). The
documentation lists no default storage format per channel and no labels for the Add channel menu: those
are ours, after the application's screenshots (Base color, short "color" on the material buttons…).
"""

from __future__ import annotations

from dataclasses import dataclass

from trafilatura3d import shaders

# Texture Set settings > Channel Settings > format: (gray, depth for the evaluator, HDR). sRGB8 stores
# gamma corrected values; the format never changes how a channel is interpreted (documentation).
FORMATS = {
    "sRGB8": (False, "8", False),
    "L8": (True, "8", False),
    "RGB8": (False, "8", False),
    "L16": (True, "16", False),
    "RGB16": (False, "16", False),
    "L16F": (True, "16f", True),
    "RGB16F": (False, "16f", True),
    "L32F": (True, "32f", True),
    "RGB32F": (False, "32f", True),
}


@dataclass(frozen=True)
class ChannelDef:
    id: str  # the shader API tag (channel_<id>)
    label: str
    short: str  # material buttons of the Properties (color, height, rough…)
    gray: bool
    color: bool  # color managed (the documentation: Base color, Emissive, Diffuse, Specular…)
    default: tuple  # lib-sampler DEFAULT_* (RGB or gray), what the shader reads without the channel
    format: str  # ours: the documentation gives no default format


def _c(id, label, short, gray, color, default, fmt):
    return ChannelDef(id, label, short, gray, color, tuple(float(x) for x in default), fmt)


# Shader API order (all-engine-params). Defaults from lib-sampler where it gives one, else ours.
CHANNELS: dict[str, ChannelDef] = {c.id: c for c in [
    _c("ambientocclusion", "Ambient occlusion", "ao", True, False, (1.0,), "L8"),
    _c("anisotropyangle", "Anisotropy angle", "aniso angle", True, False, (0.0,), "L8"),
    _c("anisotropylevel", "Anisotropy level", "aniso level", True, False, (0.0,), "L8"),
    _c("basecolor", "Base color", "color", False, True, (0.5, 0.5, 0.5), "sRGB8"),
    _c("blendingmask", "Blending mask", "mask", True, False, (0.0,), "L8"),
    _c("diffuse", "Diffuse", "diff", False, True, (0.5, 0.5, 0.5), "sRGB8"),
    _c("displacement", "Displacement", "disp", True, False, (0.0,), "L16F"),
    _c("emissive", "Emissive", "emiss", False, True, (0.0, 0.0, 0.0), "sRGB8"),
    _c("glossiness", "Glossiness", "gloss", True, False, (0.7,), "L8"),
    _c("height", "Height", "height", True, False, (0.0,), "L16F"),
    _c("ior", "IOR", "ior", True, False, (1.5,), "L16F"),
    _c("metallic", "Metallic", "metal", True, False, (0.0,), "L8"),
    _c("normal", "Normal", "nrm", False, False, (0.5, 0.5, 1.0), "RGB16F"),
    _c("opacity", "Opacity", "op", True, False, (1.0,), "L8"),
    _c("reflection", "Reflection", "refl", False, True, (0.0, 0.0, 0.0), "sRGB8"),
    _c("roughness", "Roughness", "rough", True, False, (0.3,), "L8"),
    _c("scattering", "Scattering", "scatt", True, False, (0.0,), "L8"),
    _c("specular", "Specular", "spec", False, True, (0.04, 0.04, 0.04), "sRGB8"),
    _c("specularlevel", "Specular level", "spec level", True, False, (0.5,), "L8"),
    _c("transmissive", "Transmissive", "trsl", True, False, (0.0,), "L8"),
]}
USER_CHANNELS = [f"user{i}" for i in range(8)]  # shader API: channel_user0..7
for _u in USER_CHANNELS:
    CHANNELS[_u] = ChannelDef(_u, _u.capitalize(), _u, False, False, (0.0, 0.0, 0.0), "RGB8")



def channel_id(usage: str) -> str:
    """The Texture Set channel of a shader channel (a graph output's usage): its name in lower case."""
    return usage.lower()


def _words(usage: str) -> str:
    return "".join(f" {c.lower()}" if c.isupper() else c for c in usage).capitalize()


# The other shaders' channels (shaders.TABLE: OpenPBR's specular roughness, the coats…) as channels too, from their
# renderer slot: one value or a color, its neutral value, sRGB8 for colors (normals RGB16F), L8 for the rest (ours).
for _s in shaders.TABLE:
    for _usage, _slot in _s.channels.items():
        _id = channel_id(_usage)
        if _id not in CHANNELS:
            _rgba, _gray = shaders.SLOTS[_slot]
            _normal = "normal" in _id
            CHANNELS[_id] = ChannelDef(_id, _words(_usage), _words(_usage).lower(), _gray, not _gray and not _normal,
                                       _rgba[:1] if _gray else _rgba[:3],
                                       "L8" if _gray else "RGB16F" if _normal else "sRGB8")


def render_slots(shader_name: str) -> dict:
    """Texture Set channel -> renderer slot (shaders.SLOTS) of what the shader reads: the viewport's material."""
    return {channel_id(u): slot for u, slot in shaders.BY_NAME[shader_name].channels.items()}


def supported(shader_name: str) -> tuple:
    """The "Supported channels" of the Add channel menu: the ones the project's shader reads (the others are kept
    and exported, not rendered: the "Unsupported channels", and the User channels)."""
    return tuple(render_slots(shader_name))


def unsupported(shader_name: str) -> tuple:
    return tuple(c for c in CHANNELS if c not in supported(shader_name) and c not in USER_CHANNELS)

# Default blending of a channel in new layers (layer stack documentation: Height Linear Dodge (Add),
# Normal Normal Map Detail, Ambient occlusion Multiply; the rest Normal).
DEFAULT_BLEND = {"height": "lineardodge", "normal": "normaldetail", "ambientocclusion": "multiply"}


def is_gray(channel: str, fmt: str) -> bool:
    return FORMATS[fmt][0]


def depth(fmt: str) -> str:
    return FORMATS[fmt][1]


def is_hdr(fmt: str) -> bool:
    return FORMATS[fmt][2]


def default_rgba(channel: str) -> tuple:
    d = CHANNELS[channel].default
    return (d[0], d[0], d[0], 0.0) if len(d) == 1 else (*d, 0.0)
