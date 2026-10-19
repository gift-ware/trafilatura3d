"""The 3D renderers' shaders: the one table every view reads (the Graph workspace's 3D View materials, the Paint
Shader settings, the rasterizer and the path tracer).

Shaders: OpenPBR (the default), ASM and UsdPreviewSurface for the Rasterizer / GPU Pathtracer, then Blinn, Lambert,
Mesh Info, Metallic roughness, - Coated, - SSS, Specular glossiness, Unlit; legacy usages are matched to OpenPBR.
Specifications: OpenPBR Surface (ASWF), ASM 1.0 (usages, defaults), UsdPreviewSurface (openusd.org). Not offered:
AxF SVBRDF (proprietary format).

Each shader lists its channels (graph output usage -> renderer slot, `channels`) and its parameters (`params`, the
Properties rows: a slot's value when no texture feeds it, or a uniform of the material). Where the documentation names
no usage or parameter the entry says "ours". A slot holds the channel as the graph produces it; the WGSL `lobes()` of
render3d turns it into the common layered model per shader (glossiness inverted, specular level doubled…).

Transparency: every material has a blending mode (`BLEND_MODES`, the uniform `blending`), what the renderers do with
the Opacity channel. Opaque ignores it; Alpha test discards below the Alpha threshold (lib-alpha: "Alpha
threshold" 0.33, Common Parameters); Alpha blend blends it over what is behind (render state `blend over`).
The Rasterizer blends the opacity ("Rasterizer order independent transparency enabled" chooses how), so the shaders
with an Opacity channel default to Alpha blend and the "Blending mode" / "Alpha threshold" rows are ours.
UsdPreviewSurface has none: its specification decides by opacityThreshold (above 0 a cutout at that threshold, 0
translucent).

Templates (`TEMPLATES`, ours): one per shader with channels and per blending mode it offers; a new graph gets an
Output per channel (fed a Uniform color at the channel's neutral value) and the 3D View the shader, a new Paint project
the channels in its Texture Sets and the shader in its Shader settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Renderer slots: one texture each (a constant 1x1 when no output feeds it). name -> (default rgba, one channel)
SLOTS: dict[str, tuple[tuple, bool]] = {
    "baseColor": ((0.5, 0.5, 0.5, 1.0), False),
    "normal": ((0.5, 0.5, 1.0, 1.0), False),
    "roughness": ((0.5, 0.5, 0.5, 1.0), True),
    "metallic": ((0.0, 0.0, 0.0, 1.0), True),
    "height": ((0.5, 0.5, 0.5, 1.0), True),
    "ambientOcclusion": ((1.0, 1.0, 1.0, 1.0), True),
    "emissive": ((0.0, 0.0, 0.0, 1.0), False),
    "opacity": ((1.0, 1.0, 1.0, 1.0), True),
    "specularColor": ((1.0, 1.0, 1.0, 1.0), False),
    "specularWeight": ((1.0, 1.0, 1.0, 1.0), True),
    "anisotropyLevel": ((0.0, 0.0, 0.0, 1.0), True),
    "anisotropyAngle": ((0.0, 0.0, 0.0, 1.0), True),
    "coatWeight": ((0.0, 0.0, 0.0, 1.0), True),
    "coatColor": ((1.0, 1.0, 1.0, 1.0), False),
    "coatRoughness": ((0.0, 0.0, 0.0, 1.0), True),
    "coatNormal": ((0.5, 0.5, 1.0, 1.0), False),
    "fuzzWeight": ((0.0, 0.0, 0.0, 1.0), True),
    "fuzzColor": ((1.0, 1.0, 1.0, 1.0), False),
    "fuzzRoughness": ((0.5, 0.5, 0.5, 1.0), True),
    "subsurfaceWeight": ((0.0, 0.0, 0.0, 1.0), True),
    "subsurfaceColor": ((0.8, 0.8, 0.8, 1.0), False),
    "subsurfaceRadius": ((1.0, 1.0, 1.0, 1.0), True),
    "transmissionWeight": ((0.0, 0.0, 0.0, 1.0), True),
    "transmissionColor": ((1.0, 1.0, 1.0, 1.0), False),
}
SLOT_NAMES = list(SLOTS)

# Legacy to OpenPBR usage mappings: an output with the legacy usage feeds the OpenPBR channel.
LEGACY_TO_OPENPBR = {
    "metallic": "metalness", "specularEdgeColor": "specularColor", "roughness": "specularRoughness",
    "anisotropyLevel": "specularRoughnessAnisotropy", "IOR": "specularIOR", "absorptionColor": "transmissionColor",
    "absorptionDistance": "transmissionDepth", "translucency": "subsurfaceWeight", "scatteringColor": "subsurfaceColor",
    "scatteringDistance": "subsurfaceRadius", "scatteringDistanceScale": "subsurfaceRadiusScale",
    "coatOpacity": "coatWeight", "sheenOpacity": "fuzzWeight", "sheenColor": "fuzzColor",
    "sheenRoughness": "fuzzRoughness", "emissive": "emissionColor",
}


@dataclass(frozen=True)
class Param:
    """A row of the material's Properties. `target`: ("slot", name) = the slot's constant, ("uniform", field) = a
    field of the material uniform (render3d `M`), ("look", attribute) = a renderer setting shared by every material
    (Height level / Height scale: the Displacement window's)."""
    key: str
    label: str
    group: str
    kind: str  # float, color, bool, float2, enum
    default: object
    target: tuple
    lo: float = 0.0
    hi: float = 1.0
    choices: tuple = ()


@dataclass(frozen=True)
class Shader:
    name: str
    channels: dict  # usage -> slot (the Channels rows, in this order)
    params: tuple
    aliases: dict = field(default_factory=dict)  # other usages matched to a channel (legacy usages)


def _p(key, label, group, kind, default, target, lo=0.0, hi=1.0, choices=()):
    return Param(key, label, group, kind, default, target, lo, hi, choices)


def _slot(key, label, group, kind, default, slot, lo=0.0, hi=1.0):
    return _p(key, label, group, kind, default, ("slot", slot), lo, hi)


def _uni(key, label, group, kind, default, uniform, lo=0.0, hi=1.0, choices=()):
    return _p(key, label, group, kind, default, ("uniform", uniform), lo, hi, choices)


TILING = _uni("tiling", "Tiling", "UVs", "float", 1.0, "tiling", 0.0, 10.0)  # [0, 10] for all shaders
HEIGHT_ROWS = (_p("height_level", "Height level", "Geometry", "float", 0.5, ("look", "height_level")),
               _p("height_scale", "Height scale", "Geometry", "float", 0.05, ("look", "displacement"), 0.0, 0.5))
BLEND_MODES = ("Opaque", "Alpha test", "Alpha blend")
OPAQUE, ALPHA_TEST, ALPHA_BLEND = range(3)
ALPHA_THRESHOLD = 0.33  # the lib-alpha default


def _transparency(group):
    """The blending mode rows (ours, see the module docstring), next to the shader's Opacity row."""
    return (_uni("blending", "Blending mode", group, "enum", ALPHA_BLEND, "blending", choices=BLEND_MODES),
            _uni("alpha_threshold", "Alpha threshold", group, "float", ALPHA_THRESHOLD, "alpha_threshold"))


LEGACY_COMMON = (TILING, _uni("emissive_intensity", "Emissive intensity", "Common parameters", "float", 1.0,
                              "emission", 0.0, 10.0))  # ours: the OpenGL shaders' parameters are not documented

OPENPBR = Shader(
    "OpenPBR",
    {"baseColor": "baseColor", "metalness": "metallic", "specularWeight": "specularWeight",
     "specularColor": "specularColor", "specularRoughness": "roughness",
     "specularRoughnessAnisotropy": "anisotropyLevel", "transmissionWeight": "transmissionWeight",
     "transmissionColor": "transmissionColor", "subsurfaceWeight": "subsurfaceWeight",
     "subsurfaceColor": "subsurfaceColor", "subsurfaceRadius": "subsurfaceRadius", "coatWeight": "coatWeight",
     "coatColor": "coatColor", "coatRoughness": "coatRoughness", "fuzzWeight": "fuzzWeight", "fuzzColor": "fuzzColor",
     "fuzzRoughness": "fuzzRoughness", "emissionColor": "emissive", "opacity": "opacity", "normal": "normal",
     "coatNormal": "coatNormal", "height": "height", "ambientOcclusion": "ambientOcclusion"},
    (TILING,
     _uni("uv_scale", "UV scale", "UVs", "float2", (1.0, 1.0), "uv_scale", 0.0, 10.0),
     _uni("base_weight", "Weight", "Base", "float", 1.0, "base_weight"),
     _slot("base_color", "Color", "Base", "color", (0.8, 0.8, 0.8), "baseColor"),
     _slot("metalness", "Metalness", "Base", "float", 0.0, "metallic"),
     _uni("diffuse_roughness", "Diffuse roughness", "Base", "float", 0.0, "diffuse_roughness"),
     _slot("specular_weight", "Weight", "Specular", "float", 1.0, "specularWeight"),
     _slot("specular_color", "Color", "Specular", "color", (1.0, 1.0, 1.0), "specularColor"),
     _slot("specular_roughness", "Roughness", "Specular", "float", 0.3, "roughness"),
     _slot("specular_anisotropy", "Anisotropy", "Specular", "float", 0.0, "anisotropyLevel"),
     _uni("specular_ior", "IOR", "Specular", "float", 1.5, "ior", 1.0, 3.0),
     _slot("transmission_weight", "Weight", "Transmission", "float", 0.0, "transmissionWeight"),
     _slot("transmission_color", "Color", "Transmission", "color", (1.0, 1.0, 1.0), "transmissionColor"),
     _uni("transmission_depth", "Depth", "Transmission", "float", 0.0, "transmission_depth", 0.0, 10.0),
     _slot("subsurface_weight", "Weight", "Subsurface", "float", 0.0, "subsurfaceWeight"),
     _slot("subsurface_color", "Color", "Subsurface", "color", (0.8, 0.8, 0.8), "subsurfaceColor"),
     _slot("subsurface_radius", "Radius", "Subsurface", "float", 1.0, "subsurfaceRadius", 0.0, 10.0),
     _uni("subsurface_radius_scale", "Radius scale", "Subsurface", "color", (1.0, 0.5, 0.25), "radius_scale"),
     _slot("coat_weight", "Weight", "Coat", "float", 0.0, "coatWeight"),
     _slot("coat_color", "Color", "Coat", "color", (1.0, 1.0, 1.0), "coatColor"),
     _slot("coat_roughness", "Roughness", "Coat", "float", 0.0, "coatRoughness"),
     _uni("coat_anisotropy", "Anisotropy", "Coat", "float", 0.0, "coat_anisotropy"),
     _uni("coat_ior", "IOR", "Coat", "float", 1.6, "coat_ior", 1.0, 3.0),
     _uni("coat_darkening", "Darkening", "Coat", "float", 1.0, "coat_darkening"),
     # Fuzz Weight 0 as in the OpenPBR specification: a 1 would put a white fuzz on every material.
     _slot("fuzz_weight", "Weight", "Fuzz", "float", 0.0, "fuzzWeight"),
     _slot("fuzz_color", "Color", "Fuzz", "color", (1.0, 1.0, 1.0), "fuzzColor"),
     _slot("fuzz_roughness", "Roughness", "Fuzz", "float", 0.5, "fuzzRoughness"),
     _uni("emission_luminance", "Luminance", "Emission", "float", 0.0, "emission", 0.0, 1000.0),
     _slot("emission_color", "Color", "Emission", "color", (1.0, 0.0, 0.0), "emissive"),
     _uni("thin_film_weight", "Weight", "Thin-film", "float", 0.0, "film_weight"),
     _uni("thin_film_thickness", "Thickness", "Thin-film", "float", 0.5, "film_thickness", 0.0, 2.0),
     _uni("thin_film_ior", "IOR", "Thin-film", "float", 1.4, "film_ior", 1.0, 3.0),
     _slot("opacity", "Opacity", "Geometry", "float", 1.0, "opacity"),
     *_transparency("Geometry"),
     _uni("thin_walled", "Thin walled", "Geometry", "bool", False, "thin_walled"),
     _slot("normal", "Normal", "Geometry", "color", (0.5, 0.5, 1.0), "normal"),
     _slot("coat_normal", "Coat normal", "Geometry", "color", (0.5, 0.5, 1.0), "coatNormal"),
     _slot("height", "Height", "Geometry", "float", 0.5, "height"),
     *HEIGHT_ROWS,
     _slot("ambient_occlusion", "Ambient occlusion", "Geometry", "float", 1.0, "ambientOcclusion")),
    aliases={legacy: new for legacy, new in LEGACY_TO_OPENPBR.items()})

ASM = Shader(  # ASM specification 1.0 (model 4.0): usages and defaults as written there
    "Adobe Standard Material",
    {"baseColor": "baseColor", "roughness": "roughness", "metallic": "metallic", "opacity": "opacity",
     "specularLevel": "specularWeight", "specularEdgeColor": "specularColor", "normal": "normal", "height": "height",
     "anisotropyLevel": "anisotropyLevel", "anisotropyAngle": "anisotropyAngle", "emissive": "emissive",
     "sheenOpacity": "fuzzWeight", "sheenColor": "fuzzColor", "sheenRoughness": "fuzzRoughness",
     "translucency": "transmissionWeight", "absorptionColor": "transmissionColor",
     "scatteringColor": "subsurfaceColor", "coatOpacity": "coatWeight", "coatColor": "coatColor",
     "coatRoughness": "coatRoughness", "coatNormal": "coatNormal", "ambientOcclusion": "ambientOcclusion"},
    (TILING,
     _slot("base_color", "Base color", "Surface", "color", (0.5, 0.5, 0.5), "baseColor"),
     _slot("roughness", "Roughness", "Surface", "float", 0.3, "roughness"),
     _slot("metallic", "Metallic", "Surface", "float", 0.0, "metallic"),
     _slot("opacity", "Opacity", "Surface", "float", 1.0, "opacity"),
     *_transparency("Surface"),
     _slot("specular_level", "Specular level", "Surface", "float", 0.5, "specularWeight"),
     _slot("specular_edge_color", "Specular edge color", "Surface", "color", (1.0, 1.0, 1.0), "specularColor"),
     _uni("normal_scale", "Normal scale", "Surface detail", "float", 1.0, "normal_scale", 0.0, 4.0),
     _slot("height", "Height", "Surface detail", "float", 0.5, "height"),
     *HEIGHT_ROWS,
     _slot("anisotropy", "Anisotropy", "Anisotropy", "float", 0.0, "anisotropyLevel"),
     _slot("anisotropy_angle", "Anisotropy angle", "Anisotropy", "float", 0.0, "anisotropyAngle"),
     _uni("emission", "Emission", "Emission", "float", 0.0, "emission", 0.0, 100.0),
     _slot("emission_color", "Emission color", "Emission", "color", (1.0, 1.0, 1.0), "emissive"),
     _slot("sheen_opacity", "Sheen opacity", "Sheen", "float", 0.0, "fuzzWeight"),
     _slot("sheen_color", "Sheen color", "Sheen", "color", (1.0, 1.0, 1.0), "fuzzColor"),
     _slot("sheen_roughness", "Sheen roughness", "Sheen", "float", 0.5, "fuzzRoughness"),
     _slot("translucency", "Translucency", "Interior", "float", 0.0, "transmissionWeight"),
     _uni("ior", "Index of refraction", "Interior", "float", 1.5, "ior", 1.0, 3.0),
     _slot("absorption_color", "Absorption color", "Absorption", "color", (1.0, 1.0, 1.0), "transmissionColor"),
     _uni("absorption_distance", "Absorption distance", "Absorption", "float", 0.0, "transmission_depth", 0.0, 10.0),
     _uni("scatter", "Scatter", "Subsurface scattering", "bool", False, "scatter"),
     _slot("scattering_color", "Scattering color", "Subsurface scattering", "color", (1.0, 1.0, 1.0), "subsurfaceColor"),
     _slot("scattering_distance", "Scattering distance", "Subsurface scattering", "float", 1.0, "subsurfaceRadius",
           0.0, 10.0),
     _uni("scattering_distance_scale", "Scattering distance scale", "Subsurface scattering", "color", (1.0, 1.0, 1.0),
          "radius_scale"),
     _slot("coat_opacity", "Coat opacity", "Coat", "float", 0.0, "coatWeight"),
     _slot("coat_color", "Coat color", "Coat", "color", (1.0, 1.0, 1.0), "coatColor"),
     _slot("coat_roughness", "Coat roughness", "Coat", "float", 0.0, "coatRoughness"),
     _uni("coat_ior", "Coat index of refraction", "Coat", "float", 1.6, "coat_ior", 1.0, 3.0),
     _uni("coat_specular_level", "Coat specular level", "Coat", "float", 0.5, "coat_specular_level"),
     _uni("coat_normal_scale", "Coat normal scale", "Coat", "float", 1.0, "coat_normal_scale", 0.0, 4.0),
     _slot("ambient_occlusion", "Ambient occlusion", "Auxiliary", "float", 1.0, "ambientOcclusion")))

USD_PREVIEW = Shader(  # UsdPreviewSurface specification; the usages are ours (the legacy ones)
    "UsdPreviewSurface",
    {"baseColor": "baseColor", "emissive": "emissive", "specular": "specularColor", "metallic": "metallic",
     "roughness": "roughness", "coatOpacity": "coatWeight", "coatRoughness": "coatRoughness", "opacity": "opacity",
     "normal": "normal", "height": "height", "ambientOcclusion": "ambientOcclusion"},
    (TILING,
     _slot("diffuse_color", "diffuseColor", "Surface", "color", (0.18, 0.18, 0.18), "baseColor"),
     _slot("emissive_color", "emissiveColor", "Surface", "color", (0.0, 0.0, 0.0), "emissive"),
     _uni("use_specular_workflow", "useSpecularWorkflow", "Surface", "bool", False, "specular_workflow"),
     _slot("specular_color_usd", "specularColor", "Surface", "color", (0.0, 0.0, 0.0), "specularColor"),
     _slot("metallic", "metallic", "Surface", "float", 0.0, "metallic"),
     _slot("roughness_usd", "roughness", "Surface", "float", 0.5, "roughness"),
     _slot("clearcoat", "clearcoat", "Surface", "float", 0.0, "coatWeight"),
     _slot("clearcoat_roughness", "clearcoatRoughness", "Surface", "float", 0.01, "coatRoughness"),
     _slot("opacity", "opacity", "Surface", "float", 1.0, "opacity"),
     _uni("opacity_threshold", "opacityThreshold", "Surface", "float", 0.0, "alpha_threshold"),
     _uni("ior", "ior", "Surface", "float", 1.5, "ior", 1.0, 3.0),
     _slot("occlusion", "occlusion", "Surface", "float", 1.0, "ambientOcclusion"),
     *HEIGHT_ROWS))

MR_CHANNELS = {"baseColor": "baseColor", "normal": "normal", "roughness": "roughness", "metallic": "metallic",
               "height": "height", "ambientOcclusion": "ambientOcclusion", "emissive": "emissive", "opacity": "opacity"}
MR_PARAMS = (*LEGACY_COMMON,
             _slot("base_color", "Base color", "Values without a texture", "color", (0.5, 0.5, 0.5), "baseColor"),
             _slot("roughness", "Roughness", "Values without a texture", "float", 0.5, "roughness"),
             _slot("metallic", "Metallic", "Values without a texture", "float", 0.0, "metallic"), *HEIGHT_ROWS,
             *_transparency("Common parameters"))
METALLIC_ROUGHNESS = Shader("Metallic roughness", MR_CHANNELS, MR_PARAMS,
                            )
# "extra Coat channels": the usages are ours, ASM's coat usages.
MR_COATED = Shader(
    "Metallic roughness - Coated",
    {**MR_CHANNELS, "coatOpacity": "coatWeight", "coatColor": "coatColor", "coatRoughness": "coatRoughness",
     "coatNormal": "coatNormal"},
    (*MR_PARAMS,
     _slot("coat_opacity", "Coat opacity", "Values without a texture", "float", 1.0, "coatWeight"),
     _slot("coat_color", "Coat color", "Values without a texture", "color", (1.0, 1.0, 1.0), "coatColor"),
     _slot("coat_roughness", "Coat roughness", "Values without a texture", "float", 0.0, "coatRoughness"),
     _uni("coat_ior", "Coat IOR", "Common parameters", "float", 1.6, "coat_ior", 1.0, 3.0)))
# "extra Scattering channel": the usage is "scattering"; scattered light takes the base color (ours).
MR_SSS = Shader(
    "Metallic roughness - SSS", {**MR_CHANNELS, "scattering": "subsurfaceWeight"},
    (*MR_PARAMS,
     _slot("scattering", "Scattering", "Values without a texture", "float", 0.0, "subsurfaceWeight"),
     _slot("scattering_radius", "Scattering radius", "Common parameters", "float", 1.0, "subsurfaceRadius", 0.0, 10.0),
     _uni("scattering_color", "Scattering color", "Common parameters", "color", (1.0, 0.5, 0.25), "radius_scale")))
SPECULAR_GLOSSINESS = Shader(
    "Specular glossiness",
    {"diffuse": "baseColor", "normal": "normal", "glossiness": "roughness", "specular": "specularColor",
     "height": "height", "ambientOcclusion": "ambientOcclusion", "emissive": "emissive", "opacity": "opacity"},
    (*LEGACY_COMMON,
     _slot("diffuse", "Diffuse", "Values without a texture", "color", (0.5, 0.5, 0.5), "baseColor"),
     _slot("specular", "Specular", "Values without a texture", "color", (0.22, 0.22, 0.22), "specularColor"),
     _slot("glossiness", "Glossiness", "Values without a texture", "float", 0.5, "roughness"), *HEIGHT_ROWS,
     *_transparency("Common parameters")))
BLINN = Shader(  # "Diffuse, Specular and Glossiness channels next to standard channels like Opacity, Height and Normal"
    "Blinn",
    {"diffuse": "baseColor", "specular": "specularColor", "glossiness": "roughness", "opacity": "opacity",
     "height": "height", "normal": "normal"},
    (TILING,
     _slot("diffuse", "Diffuse", "Values without a texture", "color", (0.5, 0.5, 0.5), "baseColor"),
     _slot("specular", "Specular", "Values without a texture", "color", (0.22, 0.22, 0.22), "specularColor"),
     _slot("glossiness", "Glossiness", "Values without a texture", "float", 0.5, "roughness"), *HEIGHT_ROWS,
     *_transparency("Common parameters")))
LAMBERT = Shader("Lambert", {"diffuse": "baseColor"},
                 (TILING, _slot("diffuse", "Diffuse", "Values without a texture", "color", (0.5, 0.5, 0.5), "baseColor")))
MESH_INFO_MODES = ("Normal", "Tangent", "Binormal", "UV", "UV Tile", "Vertex color", "Position (world space)")
MESH_INFO = Shader("Mesh Info", {}, (_uni("mode", "Display", "Mesh Info", "enum", 0, "mode", choices=MESH_INFO_MODES),))
UNLIT = Shader("Unlit", {"baseColor": "baseColor"},
               (TILING, _slot("color", "Color", "Values without a texture", "color", (0.5, 0.5, 0.5), "baseColor")))

# The order (the Rasterizer's OpenPBR, ASM and UsdPreviewSurface, then the OpenGL list); the index is the
# shader id of render3d's WGSL (`M.shader`).
TABLE = (OPENPBR, ASM, USD_PREVIEW, BLINN, LAMBERT, MESH_INFO, METALLIC_ROUGHNESS, MR_COATED, MR_SSS,
         SPECULAR_GLOSSINESS, UNLIT)
SHADERS = tuple(s.name for s in TABLE)
BY_NAME = {s.name: s for s in TABLE}
DEFAULT = "OpenPBR"


def index(name: str) -> int:
    return SHADERS.index(name)


def constants(shader_name: str, values: dict) -> dict:
    """{slot: rgba} of the shader's slot parameters (what a slot shows when no texture feeds it)."""
    out = {}
    for p in BY_NAME[shader_name].params:
        if p.target[0] == "slot":
            v = values.get(p.key, p.default)
            out[p.target[1]] = (*v, 1.0) if p.kind == "color" else (float(v),) * 3 + (1.0,)
    return out


def uniforms(shader_name: str, values: dict) -> dict:
    """{uniform field: value} of the shader's uniform parameters (render3d packs the material uniform from it), with
    the blending mode of a shader without the row: by its threshold (UsdPreviewSurface), else Opaque."""
    s = BY_NAME[shader_name]
    out = {p.target[1]: values.get(p.key, p.default) for p in s.params if p.target[0] == "uniform"}
    if "blending" not in out:
        out["blending"] = ALPHA_TEST if out.get("alpha_threshold", 0.0) > 0.0 else (
            ALPHA_BLEND if "alpha_threshold" in out else OPAQUE)
    return out


def channel_of(shader_name: str, usage: str) -> str | None:
    """The channel a graph output of `usage` feeds with this shader (legacy usages matched)."""
    s = BY_NAME[shader_name]
    usage = usage if usage in s.channels else s.aliases.get(usage, usage)
    return usage if usage in s.channels else None


def blending_modes(shader_name: str) -> tuple:
    """The blending modes a shader offers (its Blending mode row), or (None,) when it has none."""
    return BLEND_MODES_OF.get(shader_name, (None,))


BLEND_MODES_OF = {s.name: tuple(range(len(BLEND_MODES))) for s in TABLE
                  if any(p.target == ("uniform", "blending") for p in s.params)}


@dataclass(frozen=True)
class Template:
    """What a new graph outputs or a new Paint project paints, and the viewport's shader for it: one per shader with
    channels and per blending mode it offers (ours). `channels`: the shader's usages in its order, without Opacity
    when opaque (nothing would read it)."""
    name: str
    shader: str
    blending: int | None  # None: the shader has no blending mode
    channels: tuple

    def values(self, alpha_threshold: float = ALPHA_THRESHOLD) -> dict:
        """The shader parameter values of the template: its blending mode and the Alpha threshold."""
        return {"alpha_threshold": alpha_threshold, **({} if self.blending is None else {"blending": self.blending})}


def _templates():
    for s in TABLE:
        if not s.channels:  # Mesh Info: nothing to output
            continue
        for mode in blending_modes(s.name):
            usages = tuple(u for u in s.channels if not (u == "opacity" and mode == OPAQUE))
            name = s.name if mode in (None, OPAQUE) else f"{s.name} - {BLEND_MODES[mode]}"
            yield Template(name, s.name, mode, usages)


TEMPLATES = {t.name: t for t in _templates()}
DEFAULT_TEMPLATE = "Metallic roughness"  # a new Paint project's, as before the templates


def channel_gray(shader_name: str, usage: str) -> bool:
    """Whether a shader channel holds one value (its renderer slot's)."""
    return SLOTS[BY_NAME[shader_name].channels[usage]][1]


def neutral(shader_name: str, usage: str) -> tuple:
    """The value of a shader channel that leaves the material as without it (its renderer slot's default)."""
    return SLOTS[BY_NAME[shader_name].channels[usage]][0]
