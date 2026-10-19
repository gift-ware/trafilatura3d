"""The 3D View's renderer: a look-dev PBR rasterizer on wgpu.

- Image based lighting from an HDR latitude/longitude environment (the HDRI nodes' convention):
  prefiltered on the GPU into a mip chain (GGX lobe widening per level), a diffuse irradiance map
  and the split-sum BRDF lookup table.
- A key light along the environment's brightest direction with a PCF shadow map, and a ground
  plane that catches its shadow and fades into the background.
- Material textures copied into mipmapped half float textures, sampled trilinear + anisotropic.
- Height as vertex displacement and as parallax occlusion mapping per pixel.
- 4x MSAA, ACES / filmic / linear tone mapping with exposure in stops, the environment (optionally
  blurred) as background.
- Scenes of several objects and materials, the shaders of `shaders.TABLE` (OpenPBR, Adobe Standard Material,
  UsdPreviewSurface, Blinn, Lambert, Mesh Info, metallic roughness and its Coated / SSS variants, specular
  glossiness, unlit) on one layered model (`SURFACE`, shared with the path tracer), the legacy ambient light and two
  point lights, a selection outline, wireframe and helper lines (grid, axis, bounding boxes, tangent frames, lights),
  HDR renders.
- Transparency by the material's blending mode (`shaders.BLEND_MODES`): Opaque ignores the opacity; Alpha test
  discards below the threshold (in the shadow map too); Alpha blend, with "Rasterizer order independent transparency
  enabled", is alpha to coverage in the opaque pass (4 samples: 5 levels, dithered, any order), else true blending
  after the opaque surfaces: no depth write, submeshes sorted back to front by the distance of their bounding box
  centre, each drawn back faces first then front faces (a convex shape blends right; a concave one, or meshes that
  interpenetrate, can blend in the wrong order where they overlap: no per pixel sorting or depth peeling). Blended
  surfaces cast a shadow dithered by their opacity (the PCF filter turns it into a partial shadow); the ground and
  the background are drawn before and stay behind them.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import wgpu

from trafilatura3d import shaders

ENV_W, ENV_H, ENV_MIPS = 1024, 512, 8
IRR_W, IRR_H = 64, 32
IRR_SRC_W, IRR_SRC_H = 128, 64  # the environment's mean per texel, summed exactly for the irradiance
LUT = 128
SHADOW = 2048
SHADOW_EXTENT = 3.0  # half size of the key light's orthographic box (meters)
SAMPLES = 4
TONEMAPS = ("ACES", "Filmic", "Linear", "OCIO view")  # OCIO view: the project's display / view
# Diagnostic mode: the documentation names the parameter but does not list the modes; these
# are ours (material channels as stored, world space normal, UVs).
DIAGNOSTICS = ("None", "Base color", "Normal (world space)", "Roughness", "Metallic", "Ambient occlusion",
               "Height", "UV")
SHADOW_MODES = ("No shadows", "Shadow map")
SLOTS = shaders.SLOT_NAMES
# Slots the shaders read one value of (.r), stored in one channel: a quarter of RGBA (`material_format`).
SINGLE = {name for name, (_, single) in shaders.SLOTS.items() if single}
EIGHT_BIT = {"r8unorm", "rg8unorm", "rgba8unorm"}
DEFAULTS = {name: default for name, (default, _) in shaders.SLOTS.items()}
# The material uniform (WGSL `M`), field -> default; a shader's uniform parameters (`shaders.uniforms`) override.
UNIFORM_FIELDS = {
    "shader": 0, "blending": 0, "mode": 0, "flags": 0,
    "tiling": 1.0, "uv_u": 1.0, "uv_v": 1.0, "emission": 1.0,
    "base_weight": 1.0, "diffuse_roughness": 0.0, "ior": 1.5, "coat_ior": 1.6,
    "coat_darkening": 1.0, "coat_anisotropy": 0.0, "film_weight": 0.0, "film_thickness": 0.5,
    "film_ior": 1.4, "transmission_depth": 0.0, "normal_scale": 1.0, "coat_normal_scale": 1.0,
    "radius_r": 1.0, "radius_g": 1.0, "radius_b": 1.0, "alpha_threshold": shaders.ALPHA_THRESHOLD,
    "coat_specular_level": 0.5, "pad0": 0.0, "pad1": 0.0, "pad2": 0.0}
FLAG_THIN, FLAG_SCATTER, FLAG_SPECULAR_WORKFLOW = 1, 2, 4
UNLIT = shaders.index("Unlit")


def blending(spec: dict) -> int:
    """The blending mode a material is drawn with (shaders.BLEND_MODES): its own, Opaque when nothing makes it see
    through (no opacity texture, the constant 1)."""
    if "opacity" not in spec and spec.get("constants", {}).get("opacity", (1.0,))[0] >= 1.0:
        return shaders.OPAQUE
    return int(spec.get("uniforms", {}).get("blending", shaders.OPAQUE))


def material_uniform(spec: dict) -> bytes:
    """The material uniform of a renderer material spec: "shader" (index in shaders.SHADERS), "uniforms" ({field:
    value} as `shaders.uniforms` gives them: uv_scale, radius_scale, thin_walled, scatter and specular_workflow
    are spread into their fields), the blending mode as drawn (`blending`)."""
    v = dict(UNIFORM_FIELDS)
    flags = 0
    for key, value in spec.get("uniforms", {}).items():
        if key == "uv_scale":
            v["uv_u"], v["uv_v"] = value
        elif key == "radius_scale":
            v["radius_r"], v["radius_g"], v["radius_b"] = value
        elif key in ("thin_walled", "scatter", "specular_workflow"):
            flags |= {"thin_walled": FLAG_THIN, "scatter": FLAG_SCATTER,
                      "specular_workflow": FLAG_SPECULAR_WORKFLOW}[key] if value else 0
        else:
            v[key] = value
    v["shader"], v["blending"], v["flags"] = spec.get("shader", 0), blending(spec), flags
    return struct.pack("4I 24f", int(v["shader"]), int(v["blending"]), int(v["mode"]), int(v["flags"]),
                       *(float(v[k]) for k in list(UNIFORM_FIELDS)[4:]))

# Lat-long mapping identical to the HDRI nodes (lib.wgsl ll_dir / dir_ll): Y up, the centre of the
# image looks along -Z.
COMMON = """
const PI = 3.14159265358979;
const TAU = 6.28318530717959;
fn dir_to_uv(n: vec3f) -> vec2f {
    return vec2f(atan2(n.x, -n.z) / TAU + 0.5, 0.5 - asin(clamp(n.y, -1.0, 1.0)) / PI);
}
fn uv_to_dir(uv: vec2f) -> vec3f {
    let lon = (uv.x - 0.5) * TAU;
    let lat = (0.5 - uv.y) * PI;
    return vec3f(cos(lat) * sin(lon), sin(lat), -cos(lat) * cos(lon));
}
fn hammersley(i: u32, n: u32) -> vec2f {
    return vec2f(f32(i) / f32(n), f32(reverseBits(i)) * 2.3283064365386963e-10);
}
fn basis(n: vec3f) -> mat3x3f {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let t = normalize(cross(up, n));
    return mat3x3f(t, cross(n, t), n);
}
fn ggx_half(xi: vec2f, n: vec3f, a: f32) -> vec3f {
    let phi = TAU * xi.x;
    let ct = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let st = sqrt(1.0 - ct * ct);
    return normalize(basis(n) * vec3f(cos(phi) * st, sin(phi) * st, ct));
}
fn inside(c: vec2i, d: vec2i) -> bool { return c.x < d.x && c.y < d.y; }
fn charlie(nh: f32, r: f32) -> f32 {  // the sheen NDF of Estevez and Kulla, roughness r^2
    let inv = 1.0 / max(r * r, 1e-3);
    return (2.0 + inv) * pow(max(1.0 - nh * nh, 0.0), inv * 0.5) / TAU;
}
fn neubelt(nl: f32, nv: f32) -> f32 { return 1.0 / max(4.0 * (nl + nv - nl * nv), 1e-4); }
"""

# The shaders' common surface (both renderers): the material uniform `M`, the raw channels `Ch` (one per renderer
# slot, as the graph outputs them), `lobes()` = each shader of shaders.TABLE mapped onto one layered model, the
# OpenPBR structure of the specification's "reduction to a mixture of lobes":
#   surface = F fuzz + (1 - F E_fuzz) coated;  coated = C coat + lerp(1, T_coat (1 - E_coat), C) Delta base
#   base = lerp(dielectric, metal, M);  dielectric = specular + (1 - E_spec) lerp(lerp(diffuse, sss, S), trans, T)
# and `shade_light()`, the reflection of one light (the rasterizer's lights, the path tracer's BSDF evaluation).
# Approximations, all ours where the specification leaves the implementation free:
# - specular: GGX with the OpenPBR anisotropy mapping, height correlated Smith, multiple scattering compensation
#   (Fdez-Aguera: 1 + F0 (1 / E_ss - 1)); dielectric Fresnel exact with the specular_weight modulated IOR, metal F82-tint;
# - diffuse: Fujii's Oren-Nayar (no energy compensation term);
# - coat: GGX lobe with its own normal, the base roughened, coat IOR in the base's ratio (with the TIR fix),
#   view dependent absorption and the darkening factor (1 - K) / (1 - E_b K) with K = lerp(Ks, Kr, r_b);
# - fuzz: the Charlie sheen NDF with Neubelt's visibility in place of the specification's LTC table (not public);
#   its albedo is integrated into the BRDF lookup table (blue);
# - thin film: two-beam Airy interference at 650/510/475 nm with normal incidence amplitudes (the base taken as a
#   dielectric of IOR from its F0, so no metal phase), blended by the film weight; not Belcour-Barla;
# - subsurface: wrapped diffuse per channel (wrap = radius / (1 + radius)) in subsurface_color, no light transport;
# - transmission (rasterizer): the environment along the refracted ray, absorbed as transmission_color^(1 / depth)
#   over one scene unit (constant tint at depth 0); no scene behind, no dispersion, no volume scattering.
SURFACE = """
struct M {
    shader: u32, blending: u32, mode: u32, flags: u32,  // blending: shaders.BLEND_MODES
    tiling: f32, uv_u: f32, uv_v: f32, emission: f32,
    base_weight: f32, diffuse_roughness: f32, ior: f32, coat_ior: f32,
    coat_darkening: f32, coat_anisotropy: f32, film_weight: f32, film_thickness: f32,
    film_ior: f32, transmission_depth: f32, normal_scale: f32, coat_normal_scale: f32,
    radius_r: f32, radius_g: f32, radius_b: f32, alpha_threshold: f32,
    coat_specular_level: f32, pad0: f32, pad1: f32, pad2: f32,
};
const ALPHA_TEST = 1u;   // shaders.BLEND_MODES (0 opaque)
const ALPHA_BLEND = 2u;
const FLAG_THIN = 1u;
const FLAG_SCATTER = 2u;
const FLAG_SPECULAR_WORKFLOW = 4u;
const MODEL_LAYERED = 0u;
const MODEL_BLINN = 1u;
const MODEL_LAMBERT = 2u;
const MODEL_UNLIT = 3u;
const MODEL_MESH_INFO = 4u;

// The channels as sampled (colors sRGB encoded, normals tangent space 0..1), in the path tracer's pack order.
struct Ch {
    base: vec3f, opacity: f32, nrm: vec3f, height: f32,
    rough: f32, metal: f32, ao: f32, spec_weight: f32,
    emit: vec3f, aniso: f32, spec_color: vec3f, aniso_angle: f32,
    coat_color: vec3f, coat: f32, coat_nrm: vec3f, coat_rough: f32,
    fuzz_color: vec3f, fuzz: f32, sss_color: vec3f, sss: f32,
    trans_color: vec3f, trans: f32, fuzz_rough: f32, sss_radius: f32,
};

struct Lobes {
    model: u32,
    albedo: vec3f, metal: f32,
    f0m: vec3f, metal_weight: f32,  // metal: F0 (base weight x color) and its specular weight
    tint: vec3f, eta: f32,          // specular color (dielectric tint, metal edge), relative IOR of the dielectric
    f0d: vec3f, direct: f32,        // direct > 0: the dielectric F0 is given (specular workflows), F90 white
    rough: f32, aniso: f32, angle: f32, diffuse_rough: f32,
    coat: f32, coat_tint: vec3f, coat_rough: f32, coat_aniso: f32, coat_eta: f32, coat_scale: f32,
    darkening: f32,
    fuzz: f32, fuzz_color: vec3f, fuzz_rough: f32,
    sss: f32, sss_color: vec3f, radius: vec3f,
    trans: f32, trans_color: vec3f, depth: f32, thin: f32,
    film: f32, film_d: f32, film_ior: f32,
    emission: vec3f, opacity: f32, ao: f32, gloss_exp: f32,
};

// Shading frame: base and coat normals, the anisotropy tangent and bitangent, the geometric normal.
struct Frame { n: vec3f, cn: vec3f, t: vec3f, b: vec3f, ng: vec3f };

fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn lin(c: vec3f) -> vec3f { return srgb_to_linear(clamp(c, vec3f(0.0), vec3f(1.0))); }
fn lum3(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// OpenPBR: specular_weight scales the dielectric F0 by changing the IOR ratio; the coat changes the outside IOR.
fn modulated_eta(nb: f32, nc: f32, coat: f32, weight: f32) -> f32 {
    var under = nb / nc;
    if (nc > nb) { under = nc / nb; }  // the specification's fix of the spurious total internal reflection
    let eta = mix(nb, under, clamp(coat, 0.0, 1.0));
    let fs = pow((1.0 - eta) / (1.0 + eta), 2.0);
    let xi = clamp(weight, 0.0, 1.0 / max(fs, 1e-6));
    let e = sign(eta - 1.0) * sqrt(xi * fs);
    return (1.0 + e) / max(1.0 - e, 1e-4);
}

fn lobes(m: M, c: Ch) -> Lobes {
    var L: Lobes;
    L.model = MODEL_LAYERED;
    L.albedo = lin(c.base);
    L.metal = 0.0; L.f0m = L.albedo; L.metal_weight = 1.0;
    L.tint = vec3f(1.0); L.eta = 1.5; L.f0d = vec3f(0.0); L.direct = 0.0;
    L.rough = clamp(c.rough, 0.0, 1.0); L.aniso = 0.0; L.angle = 0.0; L.diffuse_rough = 0.0;
    L.coat = 0.0; L.coat_tint = vec3f(1.0); L.coat_rough = 0.0; L.coat_aniso = 0.0; L.coat_eta = m.coat_ior;
    L.coat_scale = 1.0; L.darkening = 1.0;
    L.fuzz = 0.0; L.fuzz_color = vec3f(1.0); L.fuzz_rough = 0.5;
    L.sss = 0.0; L.sss_color = L.albedo; L.radius = vec3f(0.0);
    L.trans = 0.0; L.trans_color = vec3f(1.0); L.depth = 0.0; L.thin = 0.0;
    L.film = 0.0; L.film_d = 0.5; L.film_ior = 1.4;
    L.emission = lin(c.emit) * m.emission; L.opacity = clamp(c.opacity, 0.0, 1.0); L.ao = clamp(c.ao, 0.0, 1.0);
    L.gloss_exp = 1.0;
    let radius = vec3f(m.radius_r, m.radius_g, m.radius_b) * max(c.sss_radius, 0.0);
    switch m.shader {
        case 0u: {  // OpenPBR
            L.albedo = lin(c.base) * m.base_weight;
            L.f0m = L.albedo;
            L.metal = c.metal;
            L.metal_weight = c.spec_weight;
            L.tint = lin(c.spec_color);
            L.aniso = c.aniso;
            L.diffuse_rough = m.diffuse_roughness;
            L.coat = c.coat; L.coat_tint = lin(c.coat_color); L.coat_rough = c.coat_rough;
            L.coat_aniso = m.coat_anisotropy; L.darkening = m.coat_darkening;
            L.eta = modulated_eta(m.ior, m.coat_ior, c.coat, c.spec_weight);
            L.fuzz = c.fuzz; L.fuzz_color = lin(c.fuzz_color); L.fuzz_rough = c.fuzz_rough;
            L.sss = c.sss; L.sss_color = lin(c.sss_color); L.radius = radius;
            L.trans = c.trans; L.trans_color = lin(c.trans_color); L.depth = m.transmission_depth;
            L.thin = select(0.0, 1.0, (m.flags & FLAG_THIN) != 0u);
            L.film = m.film_weight; L.film_d = m.film_thickness; L.film_ior = m.film_ior;
            L.emission = lin(c.emit) * m.emission * 0.01;  // luminance in nits: 100 nits = 1 (ours)
        }
        case 1u: {  // Adobe Standard Material: specular level 0.5 = the IOR's reflectivity (linear), metals unaffected
            L.metal = c.metal;
            L.tint = lin(c.spec_color);
            L.eta = modulated_eta(m.ior, m.coat_ior, c.coat, 2.0 * c.spec_weight);
            L.aniso = c.aniso; L.angle = c.aniso_angle * TAU;
            L.coat = c.coat; L.coat_tint = lin(c.coat_color); L.coat_rough = c.coat_rough;
            L.coat_scale = 2.0 * m.coat_specular_level;
            L.fuzz = c.fuzz; L.fuzz_color = lin(c.fuzz_color); L.fuzz_rough = c.fuzz_rough;
            if ((m.flags & FLAG_SCATTER) != 0u) {  // scattering instead of clear refraction
                L.sss = c.trans; L.sss_color = lin(c.sss_color); L.radius = radius;
            } else {
                L.trans = c.trans; L.trans_color = lin(c.trans_color); L.depth = m.transmission_depth;
            }
        }
        case 2u: {  // UsdPreviewSurface: metalness or specular workflow, a white clearcoat lobe with the same IOR
            if ((m.flags & FLAG_SPECULAR_WORKFLOW) != 0u) {
                L.direct = 1.0; L.f0d = lin(c.spec_color);
            } else {
                L.metal = c.metal; L.eta = m.ior;
            }
            L.coat = c.coat; L.coat_rough = c.coat_rough; L.coat_eta = m.ior; L.darkening = 0.0;
            L.emission = lin(c.emit);
        }
        case 3u: {  // Blinn (not PBR): exponent 2^(13 glossiness), ours
            L.model = MODEL_BLINN;
            L.tint = lin(c.spec_color);
            L.gloss_exp = exp2(13.0 * clamp(c.rough, 0.0, 1.0));
            L.rough = pow(2.0 / (L.gloss_exp + 2.0), 0.25);  // the GGX lobe of the same width, for the environment
        }
        case 4u: { L.model = MODEL_LAMBERT; }
        case 5u: { L.model = MODEL_MESH_INFO; }
        case 6u: { L.metal = c.metal; }  // Metallic roughness: dielectric of IOR 1.5 (F0 0.04)
        case 7u: {  // Metallic roughness - Coated
            L.metal = c.metal;
            L.coat = c.coat; L.coat_tint = lin(c.coat_color); L.coat_rough = c.coat_rough;
            L.eta = modulated_eta(1.5, m.coat_ior, c.coat, 1.0);
        }
        case 8u: {  // Metallic roughness - SSS: the scattering channel = subsurface weight, in the base color
            L.metal = c.metal;
            L.sss = c.sss; L.sss_color = L.albedo; L.radius = radius;
        }
        case 9u: {  // Specular glossiness
            L.direct = 1.0; L.f0d = lin(c.spec_color); L.rough = 1.0 - clamp(c.rough, 0.0, 1.0);
        }
        default: { L.model = MODEL_UNLIT; }
    }
    L.metal = clamp(L.metal, 0.0, 1.0);
    L.coat = clamp(L.coat, 0.0, 1.0);
    L.fuzz = clamp(L.fuzz, 0.0, 1.0);
    L.sss = clamp(L.sss, 0.0, 1.0);
    L.trans = clamp(L.trans, 0.0, 1.0);
    return L;
}

// --- Fresnel -----------------------------------------------------------------------------------------------------
fn fresnel_dielectric(mu: f32, eta: f32) -> f32 {
    let c = clamp(abs(mu), 0.0, 1.0);
    var g = eta * eta - 1.0 + c * c;
    if (g < 0.0) { return 1.0; }
    g = sqrt(g);
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return clamp(0.5 * a * a * (1.0 + b * b), 0.0, 1.0);
}
fn schlick3(f0: vec3f, f90: vec3f, mu: f32) -> vec3f { return f0 + (f90 - f0) * pow(1.0 - clamp(mu, 0.0, 1.0), 5.0); }
fn f82(f0: vec3f, tint: vec3f, mu: f32) -> vec3f {  // OpenPBR's metal: F82-tint
    let mb = 1.0 / 7.0;
    let fb = schlick3(f0, vec3f(1.0), mb);
    let k = mu * pow(1.0 - mu, 6.0) / (mb * pow(1.0 - mb, 6.0));
    return clamp(schlick3(f0, vec3f(1.0), mu) - k * (fb - tint * fb), vec3f(0.0), vec3f(1.0));
}
fn film_reflectance(mu: f32, nf: f32, d_um: f32, nb: vec3f) -> vec3f {
    let cos_f = sqrt(max(1.0 - (1.0 - mu * mu) / (nf * nf), 0.0));
    let r12 = (1.0 - nf) / (1.0 + nf);
    let r23 = (vec3f(nf) - nb) / (vec3f(nf) + nb);
    let phi = 4.0 * PI * nf * d_um * cos_f / vec3f(0.65, 0.51, 0.475);
    let x = 2.0 * r12 * r23 * cos(phi);
    let r0 = clamp((r12 * r12 + r23 * r23 + x) / (1.0 + r12 * r12 * r23 * r23 + x), vec3f(0.0), vec3f(1.0));
    return schlick3(r0, vec3f(1.0), mu);
}
fn spec_fresnel(L: Lobes, mu: f32) -> vec3f {  // the dielectric's reflection, tinted
    var f: vec3f;
    var nb = vec3f(L.eta);
    if (L.direct > 0.0) {
        f = schlick3(L.f0d, vec3f(1.0), mu);
        let s = sqrt(clamp(L.f0d, vec3f(0.0), vec3f(0.99)));
        nb = (1.0 + s) / (1.0 - s);
    } else {
        f = L.tint * fresnel_dielectric(mu, L.eta);
    }
    if (L.film > 0.0) { f = mix(f, film_reflectance(mu, L.film_ior, L.film_d, nb), L.film); }
    return f;
}
fn metal_fresnel(L: Lobes, mu: f32) -> vec3f {
    var f = f82(L.f0m, L.tint, mu);
    if (L.film > 0.0) {
        let s = sqrt(clamp(L.f0m, vec3f(0.0), vec3f(0.99)));
        f = mix(f, film_reflectance(mu, L.film_ior, L.film_d, (1.0 + s) / (1.0 - s)), L.film);
    }
    return f * L.metal_weight;
}
fn coat_fresnel(L: Lobes, mu: f32) -> f32 { return clamp(fresnel_dielectric(mu, L.coat_eta) * L.coat_scale, 0.0, 1.0); }

// --- microfacet ----------------------------------------------------------------------------------------------------
fn alphas(r: f32, a: f32) -> vec2f {  // OpenPBR: at = r^2 sqrt(2 / (1 + (1 - a)^2)), ab = (1 - a) at
    let r2 = max(r, 0.03) * max(r, 0.03);
    let at = r2 * sqrt(2.0 / (1.0 + (1.0 - a) * (1.0 - a)));
    return vec2f(max(at, 1e-3), max((1.0 - a) * at, 1e-3));
}
fn ggx_aniso(nh: f32, th: f32, bh: f32, a: vec2f) -> f32 {
    let x = th / a.x;
    let y = bh / a.y;
    let d = x * x + y * y + nh * nh;
    return 1.0 / (PI * a.x * a.y * d * d);
}
// Height correlated Smith visibility G / (4 nl nv), anisotropic.
fn smith_vis(nl: f32, nv: f32, tl: f32, bl: f32, tv: f32, bv: f32, a: vec2f) -> f32 {
    let gv = nl * length(vec3f(a.x * tv, a.y * bv, nv));
    let gl = nv * length(vec3f(a.x * tl, a.y * bl, nl));
    return 0.5 / max(gv + gl, 1e-5);
}
// The BRDF lookup table at (cos, roughness), kept inside its texel centres: the sampler repeats in u.
fn brdf_lut(nv: f32, r: f32) -> vec4f {
    let half = 0.5 / f32(textureDimensions(brdf).x);
    return textureSampleLevel(brdf, lin_samp, clamp(vec2f(nv, r), vec2f(half), vec2f(1.0 - half)), 0.0);
}
// Directional albedo of a Schlick lobe (split sum lookup table: red F0 weight, green F90 weight).
fn spec_albedo(f0: vec3f, f90: vec3f, nv: f32, r: f32) -> vec3f {
    let ab = brdf_lut(nv, r).rg;
    return f0 * ab.x + f90 * ab.y;
}
fn ms_comp(f0: vec3f, nv: f32, r: f32) -> vec3f {
    let ab = brdf_lut(nv, r).rg;
    return 1.0 + f0 * (1.0 / max(ab.x + ab.y, 1e-3) - 1.0);
}
fn fuzz_albedo(nv: f32, r: f32) -> f32 { return brdf_lut(nv, r).b; }
fn coated_roughness(L: Lobes) -> f32 {  // the base seen through a rough coat
    let r4 = pow(L.rough, 4.0) + 2.0 * pow(L.coat_rough, 4.0);
    return mix(L.rough, pow(min(1.0, r4), 0.25), L.coat);
}
// Coat darkening (specification: Delta = (1 - K) / (1 - E_b K)), modulated by coat weight and darkening.
fn darkening(L: Lobes, nv: f32) -> f32 {
    if (L.coat <= 0.0 || L.darkening <= 0.0) { return 1.0; }
    let ec = L.coat_eta;
    let favg = clamp((ec - 1.0) / (4.08567 + 1.00071 * ec), 0.0, 1.0);
    let kr = 1.0 - (1.0 - favg) / (ec * ec);
    let ks = fresnel_dielectric(nv, ec);
    let fs = pow((1.0 - L.eta) / (1.0 + L.eta), 2.0);
    let rb = mix(mix(1.0, L.rough, clamp(fs, 0.0, 1.0)), L.rough, L.metal);
    let k = mix(ks, kr, rb);
    let eb = mix(lum3(L.albedo) * (1.0 - L.trans), lum3(L.f0m), L.metal);
    let delta = (1.0 - k) / max(1.0 - eb * k, 1e-4);
    return mix(1.0, delta, L.coat * L.darkening);
}
fn coat_transmittance(L: Lobes, mu: f32) -> vec3f {  // view dependent absorption, coat_color at normal incidence
    let mt = sqrt(max(1.0 - (1.0 - mu * mu) / (L.coat_eta * L.coat_eta), 1e-4));
    return pow(max(L.coat_tint, vec3f(1e-4)), vec3f(1.0 / mt));
}
fn wrap(nl: f32, w: vec3f) -> vec3f { return max((vec3f(nl) + w) / ((1.0 + w) * (1.0 + w)), vec3f(0.0)); }
fn sss_wrap(L: Lobes) -> vec3f { return clamp(L.radius / (1.0 + L.radius), vec3f(0.0), vec3f(0.9)); }

// Fujii's Oren-Nayar, sigma = diffuse roughness (OpenPBR), / PI included.
fn oren_nayar(nl: f32, nv: f32, lv: f32, sigma: f32) -> f32 {
    let s = lv - nl * nv;
    let t = select(1.0, max(nl, nv), s > 0.0);
    let a = 1.0 / (1.0 + (PI * 0.5 - 2.0 / 3.0) * sigma);
    return (a + sigma * a * s / t) / PI;
}

// The reflection of light from `l` with `radiance`, cosine included: f(v, l) cos.
fn shade_light(L: Lobes, fr: Frame, v: vec3f, l: vec3f, radiance: vec3f) -> vec3f {
    let n = fr.n;
    let nl = dot(n, l);
    let nv = max(dot(n, v), 1e-4);
    if (L.model == MODEL_LAMBERT) { return L.albedo / PI * max(nl, 0.0) * radiance; }
    let h = normalize(l + v);
    let nh = max(dot(n, h), 0.0);
    if (L.model == MODEL_BLINN) {
        if (nl <= 0.0) { return vec3f(0.0); }
        let spec = L.tint * (L.gloss_exp + 8.0) / (8.0 * PI) * pow(nh, L.gloss_exp);
        return (L.albedo / PI + spec) * nl * radiance;
    }
    if (L.model != MODEL_LAYERED) { return vec3f(0.0); }
    let vh = max(dot(v, h), 0.0);
    let rb = coated_roughness(L);
    let e_d = spec_albedo(spec_fresnel(L, 1.0), select(vec3f(1.0), L.tint, L.direct == 0.0), nv, rb);
    let opaque = (1.0 - L.trans) * (1.0 - L.metal);
    var base = vec3f(0.0);
    if (nl > 0.0) {
        let a = alphas(rb, L.aniso);
        let dv = ggx_aniso(nh, dot(fr.t, h), dot(fr.b, h), a)
               * smith_vis(nl, nv, dot(fr.t, l), dot(fr.b, l), dot(fr.t, v), dot(fr.b, v), a);
        let spec_d = dv * spec_fresnel(L, vh) * ms_comp(spec_fresnel(L, 1.0), nv, rb);
        let spec_m = dv * metal_fresnel(L, vh) * ms_comp(L.f0m, nv, rb);
        let diffuse = L.albedo * oren_nayar(nl, nv, dot(l, v), L.diffuse_rough);
        base = (mix(spec_d, spec_m, L.metal) + opaque * (1.0 - L.sss) * (1.0 - e_d) * diffuse) * nl;
    }
    if (L.sss > 0.0) { base += opaque * L.sss * (1.0 - e_d) * L.sss_color / PI * wrap(nl, sss_wrap(L)); }
    var surface = base;
    if (L.coat > 0.0) {
        let cn = fr.cn;
        let cl = dot(cn, l);
        let cv = max(dot(cn, v), 1e-4);
        var coat = vec3f(0.0);
        if (cl > 0.0) {
            let hc = normalize(l + v);
            let ac = alphas(L.coat_rough, L.coat_aniso);
            coat = vec3f(ggx_aniso(max(dot(cn, hc), 0.0), dot(fr.t, hc), dot(fr.b, hc), ac)
                 * smith_vis(cl, cv, dot(fr.t, l), dot(fr.b, l), dot(fr.t, v), dot(fr.b, v), ac)
                 * coat_fresnel(L, max(dot(v, hc), 0.0)) * cl);
        }
        let ec = spec_albedo(vec3f(coat_fresnel(L, 1.0)), vec3f(L.coat_scale), cv, L.coat_rough).x;
        let t = coat_transmittance(L, cv) * coat_transmittance(L, max(cl, 1e-2));
        surface = L.coat * coat + mix(vec3f(1.0), sqrt(t) * (1.0 - ec), L.coat) * darkening(L, cv) * base;
    }
    if (L.fuzz > 0.0) {
        var fuzz = vec3f(0.0);
        if (nl > 0.0) { fuzz = L.fuzz_color * charlie(nh, L.fuzz_rough) * neubelt(nl, nv) * nl; }
        surface = L.fuzz * fuzz + (1.0 - L.fuzz * fuzz_albedo(nv, L.fuzz_rough)) * surface;
    }
    return surface * radiance;
}

// Emission through the coat (OpenPBR: lerp(1, T_coat, C)).
fn emitted(L: Lobes, nv: f32) -> vec3f {
    if (L.coat <= 0.0) { return L.emission; }
    return L.emission * mix(vec3f(1.0), coat_transmittance(L, nv), L.coat);
}

// Mesh Info: geometry data as is, clamped to [0, 1].
fn mesh_info(mode: u32, n: vec3f, t: vec3f, b: vec3f, uv: vec2f, color: vec4f, pos: vec3f) -> vec3f {
    var c: vec3f;
    switch mode {
        case 0u: { c = n; }
        case 1u: { c = t; }
        case 2u: { c = b; }
        case 3u: { c = vec3f(uv, 0.0); }
        case 4u: { c = vec3f((floor(uv) + 0.5) / 10.0, 0.0); }  // UDIM tile: tenths per tile (ours)
        case 5u: { c = color.rgb; }
        default: { c = pos; }
    }
    return clamp(c, vec3f(0.0), vec3f(1.0));
}
"""

# Any float lat-long texture -> level 0 of the environment (bilinear, wrapping in longitude).
RESAMPLE = COMMON + """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
fn at(q: vec2i, s: vec2i) -> vec4f {
    return textureLoad(src, vec2i(((q.x % s.x) + s.x) % s.x, clamp(q.y, 0, s.y - 1)), 0);
}
// Smaller than the source: the mean of the source texels the destination texel covers (a light a few texels wide
// keeps its energy; point sampling would miss it). Larger: bilinear.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(dst));
    let c = vec2i(id.xy);
    if (!inside(c, d)) { return; }
    let s = vec2i(textureDimensions(src));
    var v: vec4f;
    if (s.x > d.x || s.y > d.y) {
        let a = c * s / d;
        let b = max((c + 1) * s / d, a + 1);
        var sum = vec3f(0.0);
        for (var y = a.y; y < b.y; y++) {
            for (var x = a.x; x < b.x; x++) { sum += at(vec2i(x, y), s).rgb; }
        }
        v = vec4f(sum / f32((b.x - a.x) * (b.y - a.y)), 1.0);
    } else {
        let x = (vec2f(c) + 0.5) / vec2f(d) * vec2f(s) - 0.5;
        let i = vec2i(floor(x));
        let f = fract(x);
        v = mix(mix(at(i, s), at(i + vec2i(1, 0), s), f.x), mix(at(i + vec2i(0, 1), s), at(i + vec2i(1, 1), s), f.x), f.y);
    }
    textureStore(dst, c, vec4f(clamp(v.rgb, vec3f(0.0), vec3f(60000.0)), 1.0));
}
"""

# Level k from level k-1: widens the GGX lobe by the roughness still missing to k / (levels - 1).
PREFILTER = COMMON + """
struct P { alpha: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(dst));
    let c = vec2i(id.xy);
    if (!inside(c, d)) { return; }
    let n = uv_to_dir((vec2f(c) + 0.5) / vec2f(d));
    var acc = vec3f(0.0);
    var w = 0.0;
    for (var i = 0u; i < 128u; i++) {
        let h = ggx_half(hammersley(i, 128u), n, p.alpha);
        let l = normalize(2.0 * dot(n, h) * h - n);
        let nl = dot(n, l);
        if (nl > 0.0) {
            acc += textureSampleLevel(src, samp, dir_to_uv(l), 0.0).rgb * nl;
            w += nl;
        }
    }
    textureStore(dst, c, vec4f(acc / max(w, 1e-4), 1.0));
}
"""

IRRADIANCE = COMMON + """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
// Diffuse irradiance / PI (the mean radiance a Lambert surface sees), summed exactly over every texel of the
// environment's energy preserving reduction `src` (its own solid angle each): small bright lights keep their weight
// and their direction, so turning the environment turns the diffuse light too.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(dst));
    let c = vec2i(id.xy);
    if (!inside(c, d)) { return; }
    let n = uv_to_dir((vec2f(c) + 0.5) / vec2f(d));
    let s = vec2i(textureDimensions(src));
    let cell = TAU / f32(s.x) * PI / f32(s.y);
    var acc = vec3f(0.0);
    for (var y = 0; y < s.y; y++) {
        for (var x = 0; x < s.x; x++) {
            let l = uv_to_dir((vec2f(f32(x), f32(y)) + 0.5) / vec2f(s));
            let w = max(dot(n, l), 0.0) * sqrt(max(1.0 - l.y * l.y, 0.0));
            acc += textureLoad(src, vec2i(x, y), 0).rgb * w;
        }
    }
    textureStore(dst, c, vec4f(acc * cell / PI, 1.0));
}
"""

BRDF_LUT = COMMON + """
@group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(dst));
    let c = vec2i(id.xy);
    if (!inside(c, d)) { return; }
    let nv = (f32(c.x) + 0.5) / f32(d.x);
    let rough = (f32(c.y) + 0.5) / f32(d.y);
    let a = rough * rough;
    let v = vec3f(sqrt(1.0 - nv * nv), 0.0, nv);
    var A = 0.0;
    var B = 0.0;
    for (var i = 0u; i < 512u; i++) {
        let h = ggx_half(hammersley(i, 512u), vec3f(0.0, 0.0, 1.0), a);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let nl = l.z;
        let nh = max(h.z, 0.0);
        let vh = max(dot(v, h), 0.0);
        if (nl > 0.0) {
            let k = a * 0.5;
            let g = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
            let gv = g * vh / max(nh * nv, 1e-4);
            let fc = pow(1.0 - vh, 5.0);
            A += (1.0 - fc) * gv;
            B += fc * gv;
        }
    }
    // Blue: the directional albedo of the fuzz lobe (Charlie x Neubelt, white), uniform hemisphere samples.
    var fuzz = 0.0;
    for (var i = 0u; i < 512u; i++) {
        let xi = hammersley(i, 512u);
        let z = xi.y;
        let s = sqrt(max(1.0 - z * z, 0.0));
        let l = vec3f(cos(TAU * xi.x) * s, sin(TAU * xi.x) * s, z);
        let h = normalize(l + v);
        fuzz += charlie(h.z, rough) * neubelt(z, nv) * z * TAU;
    }
    textureStore(dst, c, vec4f(A / 512.0, B / 512.0, clamp(fuzz / 512.0, 0.0, 1.0), 1.0));
}
"""

# The default environment: a procedural photo studio (HDR) with a warm key softbox, a cool fill,
# a rim strip, a gray cyclorama and a darker floor.
STUDIO_KEY = (0.55, 0.55, 0.63)
STUDIO = COMMON + f"""
@group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;
fn softbox(r: vec3f, axis: vec3f, size: vec2f) -> f32 {{
    let q = transpose(basis(normalize(axis))) * r;
    if (q.z <= 0.0) {{ return 0.0; }}
    let p = abs(q.xy / q.z) / size;
    return 1.0 - smoothstep(0.85, 1.0, max(p.x, p.y));
}}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let d = vec2i(textureDimensions(dst));
    let c = vec2i(id.xy);
    if (!inside(c, d)) {{ return; }}
    let r = uv_to_dir((vec2f(c) + 0.5) / vec2f(d));
    // A bright neutral room (mean radiance near 1, so albedo reads as albedo) with three softboxes.
    var col = mix(vec3f(0.42, 0.41, 0.40), vec3f(0.95, 0.96, 0.98), smoothstep(-0.25, 0.3, r.y));
    col += vec3f(0.25) * smoothstep(0.5, 1.0, r.y);
    col += vec3f(7.0, 6.6, 6.0) * softbox(r, vec3f{STUDIO_KEY}, vec2f(0.32, 0.22));
    col += vec3f(2.4, 2.6, 3.0) * softbox(r, vec3f(-0.85, 0.25, 0.35), vec2f(0.45, 0.3));
    col += vec3f(4.0, 4.0, 4.0) * softbox(r, vec3f(-0.25, 0.35, -0.9), vec2f(0.06, 0.5));
    textureStore(dst, c, vec4f(col, 1.0));
}}
"""

# Material textures: level 0 converted from any storage format (gray spread to RGB), then 2x2 means; FORMAT is the
# slot's (`material_format`).
TO_HALF = """
@group(0) @binding(0) var<uniform> u: vec4u;  // gray, the rectangle's corner (x, y)
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<FORMAT, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(dst));
    let c = vec2i(id.xy) + vec2i(u.yz);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var v = textureLoad(src, c, 0);
    if (u.x != 0u) { v = vec4f(v.rrr, 1.0); }
    textureStore(dst, c, v);
}
"""
DOWNSAMPLE = """
@group(0) @binding(0) var<uniform> u: vec4u;  // the rectangle's corner (x, y)
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<FORMAT, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(dst));
    let c = vec2i(id.xy) + vec2i(u.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let m = vec2i(textureDimensions(src)) - 1;
    let q = c * 2;
    let v = textureLoad(src, min(q, m), 0) + textureLoad(src, min(q + vec2i(1, 0), m), 0)
          + textureLoad(src, min(q + vec2i(0, 1), m), 0) + textureLoad(src, min(q + vec2i(1, 1), m), 0);
    textureStore(dst, c, v * 0.25);
}
"""

SLOT_BINDINGS = "".join(f"@group(1) @binding({i + 1}) var t_{name}: texture_2d<f32>;\n" for i, name in enumerate(SLOTS))
MAT_SAMPLER = len(SLOTS) + 1

MAIN = COMMON + SURFACE + SLOT_BINDINGS + f"""@group(1) @binding({MAT_SAMPLER}) var mat: sampler;
const UNLIT = {UNLIT}u;
const MESH_INFO = {shaders.index("Mesh Info")}u;
""" + """
struct U {
    view_proj: mat4x4f, light_view_proj: mat4x4f, inv_view_proj: mat4x4f,
    eye: vec4f,
    light_dir: vec4f,    // xyz towards the key light, w intensity
    light_color: vec4f,
    params: vec4f,       // height scale, outline width, exposure (stops), parallax depth
    env: vec4f,          // rotation, intensity, background mip level, last mip level
    flags: vec4u,        // opengl normals, show environment, preview (unlit), shadows
    ground: vec4f,       // albedo, height
    tone: vec4u,         // tone mapping, diagnostic mode, color channel (0 RGB, 1 R, 2 G, 3 B), linear output
    extra: vec4f,        // ambient occlusion opacity, shadow opacity, ground shadow intensity, height level
    ambient: vec4f,      // legacy ambient light (color x intensity)
    p0_pos: vec4f, p0_col: vec4f,  // point lights: position (w enabled), color (w intensity)
    p1_pos: vec4f, p1_col: vec4f,
    wire: vec4f,         // wireframe color
    bg_top: vec4f,       // background gradient, top (sRGB), environment opacity over it
    bg_bottom: vec4f,    // background gradient, bottom (sRGB)
};
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var<uniform> m: M;
@group(2) @binding(0) var env: texture_2d<f32>;
@group(2) @binding(1) var irradiance: texture_2d<f32>;
@group(2) @binding(2) var brdf: texture_2d<f32>;
@group(2) @binding(3) var lin_samp: sampler;
@group(2) @binding(4) var shadow_map: texture_depth_2d;
@group(2) @binding(5) var shadow_cmp: sampler_comparison;
@group(2) @binding(6) var ocio_lut: texture_3d<f32>;  // working space -> display / view, log2 shaped
@group(2) @binding(7) var clamp_samp: sampler;
const LUT_LOG_MIN = -12.0;
const LUT_LOG_MAX = 8.0;
fn ocio_view(x: vec3f) -> vec3f {
    let t = clamp((log2(max(x, vec3f(1e-6))) - LUT_LOG_MIN) / (LUT_LOG_MAX - LUT_LOG_MIN), vec3f(0.0), vec3f(1.0));
    let n = f32(textureDimensions(ocio_lut).x);
    return textureSampleLevel(ocio_lut, clamp_samp, t * ((n - 1.0) / n) + 0.5 / n, 0.0).rgb;
}

var<private> gdx: vec2f;
var<private> gdy: vec2f;
// Explicit gradients: valid inside the parallax loop and any non-uniform control flow.
fn tex(t: texture_2d<f32>, uv: vec2f) -> vec4f { return textureSampleGrad(t, mat, uv, gdx, gdy); }

fn rot_y(d: vec3f) -> vec3f {
    let a = u.env.x;
    return vec3f(d.x * cos(a) - d.z * sin(a), d.y, d.x * sin(a) + d.z * cos(a));
}
fn env_at(d: vec3f, level: f32) -> vec3f {
    return textureSampleLevel(env, lin_samp, dir_to_uv(rot_y(d)), level).rgb * u.env.y;
}
fn irradiance_at(n: vec3f) -> vec3f {
    return textureSampleLevel(irradiance, lin_samp, dir_to_uv(rot_y(n)), 0.0).rgb * u.env.y;
}
fn linear_to_srgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
fn hable(x: vec3f) -> vec3f {
    let a = 0.15; let b = 0.50; let c = 0.10; let d = 0.20; let e = 0.02; let f = 0.30;
    return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
}
// A display value (sRGB encoded) through the Color channels selector.
fn channel(c: vec3f) -> vec3f {
    switch u.tone.z {
        case 1u: { return vec3f(c.r); }
        case 2u: { return vec3f(c.g); }
        case 3u: { return vec3f(c.b); }
        default: { return c; }
    }
}
// Scene linear radiance -> what is written: tone mapped sRGB, or linear with exposure (HDR renders).
fn finish(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0)) * exp2(u.params.z);
    if (u.tone.w == 1u) { return x; }
    if (u.tone.x == 3u) { return channel(ocio_view(x)); }  // the OCIO view includes its own encoding
    var mapped: vec3f;
    switch u.tone.x {
        case 1u: { mapped = hable(x * 2.0) / hable(vec3f(11.2)); }
        case 2u: { mapped = x; }
        default: { mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14); }  // ACES fit
    }
    return channel(linear_to_srgb(clamp(mapped, vec3f(0.0), vec3f(1.0))));
}
// An sRGB display value (previews, diagnostics) through the same output path.
fn display(c: vec3f) -> vec3f {
    if (u.tone.w == 1u) { return srgb_to_linear(clamp(c, vec3f(0.0), vec3f(1.0))); }
    return channel(clamp(c, vec3f(0.0), vec3f(1.0)));
}
fn shadow(world: vec3f, n: vec3f) -> f32 {
    if (u.flags.w == 0u) { return 1.0; }
    let p = u.light_view_proj * vec4f(world + n * 0.01, 1.0);
    let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z > 1.0) { return 1.0; }
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    var s = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            s += textureSampleCompareLevel(shadow_map, shadow_cmp, uv + vec2f(f32(i), f32(j)) * texel, p.z - 0.0015);
        }
    }
    return s / 25.0;
}

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f, @location(1) normal: vec3f, @location(2) tangent: vec4f, @location(3) uv: vec2f,
    @location(4) color: vec4f, @location(5) mesh_uv: vec2f,
};

fn tiled(uv: vec2f) -> vec2f { return uv * m.tiling * vec2f(m.uv_u, m.uv_v); }
fn displaced(p: vec3f, n: vec3f, uv: vec2f) -> vec3f {
    return p + n * (textureSampleLevel(t_height, mat, uv, 0.0).r - u.extra.w) * u.params.x;
}

@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) t: vec4f, @location(3) uv: vec2f,
      @location(4) color: vec4f) -> VOut {
    let tuv = tiled(uv);
    let world = displaced(p, n, tuv);
    var o: VOut;
    o.pos = u.view_proj * vec4f(world, 1.0);
    o.world = world; o.normal = n; o.tangent = t; o.uv = tuv; o.color = color; o.mesh_uv = uv;
    return o;
}

struct SOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex
fn shadow_vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) t: vec4f, @location(3) uv: vec2f) -> SOut {
    let tuv = tiled(uv);
    return SOut(u.light_view_proj * vec4f(displaced(p, n, tuv), 1.0), tuv);
}
// The shadow of a see-through material: the cutout of Alpha test; Alpha blend kept where its opacity passes the
// interleaved gradient noise of the shadow map's texel, which the PCF filter averages into a partial shadow (ours).
@fragment
fn shadow_fs(i: SOut) {
    let a = textureSample(t_opacity, mat, i.uv).r;
    if (m.blending == ALPHA_BLEND) {
        if (a <= fract(52.9829189 * fract(dot(i.pos.xy, vec2f(0.06711056, 0.00583715))))) { discard; }
    } else if (a < m.alpha_threshold) {
        discard;
    }
}

// Wireframe: the displaced edges, pulled a little towards the camera.
@vertex
fn wire_vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) t: vec4f, @location(3) uv: vec2f) -> @builtin(position) vec4f {
    var c = u.view_proj * vec4f(displaced(p, n, tiled(uv)), 1.0);
    c.z -= 2e-4 * c.w;
    return c;
}
@fragment
fn wire_fs() -> @location(0) vec4f { return u.wire; }

// Selection outline: back faces pushed out along the normal (an inverted hull), in screen space width.
@vertex
fn outline_vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) t: vec4f, @location(3) uv: vec2f) -> @builtin(position) vec4f {
    let w = displaced(p, n, tiled(uv));
    let c = u.view_proj * vec4f(w, 1.0);
    return u.view_proj * vec4f(w + n * u.params.y * c.w, 1.0);
}
@fragment
fn outline_fs() -> @location(0) vec4f { return vec4f(0.16, 0.5, 1.0, 1.0); }

// Parallax occlusion mapping: march the view ray down the height field (white = surface) in
// tangent space, then refine between the last two steps.
fn parallax(uv: vec2f, v: vec3f) -> vec2f {
    if (u.params.w <= 0.0 || v.z < 1e-3) { return uv; }
    let steps = mix(48.0, 12.0, v.z);
    let delta = -v.xy / v.z * u.params.w / steps;
    var cur = uv;
    var depth = 0.0;
    var h = 1.0 - tex(t_height, cur).r;
    var prev = vec2f(h, 0.0);
    for (var i = 0; i < 48; i++) {
        if (depth >= h || f32(i) >= steps) { break; }
        prev = vec2f(h, depth);
        cur += delta;
        depth += 1.0 / steps;
        h = 1.0 - tex(t_height, cur).r;
    }
    let after = h - depth;
    let before = prev.x - prev.y;
    let w = after / min(after - before, -1e-5);
    return cur - delta * w;
}

fn point_light(pos: vec4f, col: vec4f, world: vec3f, L: Lobes, fr: Frame, v: vec3f) -> vec3f {
    if (pos.w == 0.0) { return vec3f(0.0); }
    let d = pos.xyz - world;
    let d2 = max(dot(d, d), 1e-4);
    return shade_light(L, fr, v, d / sqrt(d2), col.rgb * col.w / d2);
}

// The slots at a texel, as the graph gave them.
fn channels(uv: vec2f) -> Ch {
    var c: Ch;
    c.base = tex(t_baseColor, uv).rgb; c.opacity = tex(t_opacity, uv).r;
    c.nrm = tex(t_normal, uv).xyz; c.height = tex(t_height, uv).r;
    c.rough = tex(t_roughness, uv).r; c.metal = tex(t_metallic, uv).r;
    c.ao = tex(t_ambientOcclusion, uv).r; c.spec_weight = tex(t_specularWeight, uv).r;
    c.emit = tex(t_emissive, uv).rgb; c.aniso = tex(t_anisotropyLevel, uv).r;
    c.spec_color = tex(t_specularColor, uv).rgb; c.aniso_angle = tex(t_anisotropyAngle, uv).r;
    c.coat_color = tex(t_coatColor, uv).rgb; c.coat = tex(t_coatWeight, uv).r;
    c.coat_nrm = tex(t_coatNormal, uv).xyz; c.coat_rough = tex(t_coatRoughness, uv).r;
    c.fuzz_color = tex(t_fuzzColor, uv).rgb; c.fuzz = tex(t_fuzzWeight, uv).r;
    c.sss_color = tex(t_subsurfaceColor, uv).rgb; c.sss = tex(t_subsurfaceWeight, uv).r;
    c.trans_color = tex(t_transmissionColor, uv).rgb; c.trans = tex(t_transmissionWeight, uv).r;
    c.fuzz_rough = tex(t_fuzzRoughness, uv).r; c.sss_radius = tex(t_subsurfaceRadius, uv).r;
    return c;
}

// A tangent space normal map texel -> world space, its effect scaled (Adobe Standard Material's normal scale: above 1
// the z axis shrinks, below 1 the angle to z shrinks).
fn mapped_normal(rgb: vec3f, scale: f32, t: vec3f, bv: vec3f, ng: vec3f) -> vec3f {
    var tn = rgb * 2.0 - 1.0;
    if (u.flags.x == 0u) { tn.y = -tn.y; }  // DirectX: green down
    if (scale > 1.0) {
        tn = normalize(vec3f(tn.xy, tn.z / scale));
    } else if (scale < 1.0) {
        tn = normalize(mix(vec3f(0.0, 0.0, 1.0), normalize(tn), max(scale, 0.0)));
    }
    return normalize(t * tn.x - bv * tn.y + ng * tn.z);  // OpenGL: green up the image (-V)
}

// Image based light of the layered model: prefiltered environment x split sum per lobe, diffuse irradiance (wrapped
// towards the back side for subsurface), the environment through the refraction for transmission.
fn shade_ibl(L: Lobes, fr: Frame, v: vec3f, ao: f32) -> vec3f {
    let n = fr.n;
    let nv = max(dot(n, v), 1e-4);
    let mips = u.env.w;
    let irr = irradiance_at(n) + u.ambient.rgb;
    if (L.model == MODEL_LAMBERT) { return vec3f(0.0); }
    let specular_ao = clamp(pow(nv + ao, exp2(-16.0 * L.rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    if (L.model == MODEL_BLINN) {
        return L.albedo * irr * ao + L.tint * env_at(reflect(-v, n), L.rough * mips) * specular_ao;
    }
    let rb = coated_roughness(L);
    var nr = n;  // anisotropy: the reflection bent along the stretched direction (Filament's bent normal)
    if (L.aniso > 0.0) {
        let at = cross(fr.b, v);
        nr = normalize(mix(n, cross(at, fr.b), L.aniso * 0.85));
    }
    let pref = env_at(reflect(-v, nr), rb * mips) * specular_ao;
    let f0d = spec_fresnel(L, 1.0);
    let spec_d = pref * spec_albedo(f0d, select(vec3f(1.0), L.tint, L.direct == 0.0), nv, rb) * ms_comp(f0d, nv, rb);
    let ratio = metal_fresnel(L, nv) / max(schlick3(L.f0m, vec3f(1.0), nv), vec3f(1e-4));  // F82 vs Schlick
    let spec_m = pref * spec_albedo(L.f0m, vec3f(1.0), nv, rb) * ms_comp(L.f0m, nv, rb) * ratio;
    let e_d = spec_albedo(f0d, select(vec3f(1.0), L.tint, L.direct == 0.0), nv, rb);
    let diffuse = L.albedo * irr * ao;
    let w = sss_wrap(L);
    let sss = L.sss_color * mix(irr, 0.5 * (irr + irradiance_at(-n)), w) * ao;
    var trans = vec3f(0.0);
    if (L.trans > 0.0) {
        var dir = -v;
        if (L.thin == 0.0) { dir = refract(-v, n, 1.0 / max(L.eta, 1.0001)); }
        var tint = L.trans_color;
        if (L.depth > 0.0) { tint = pow(max(L.trans_color, vec3f(1e-4)), vec3f(1.0 / L.depth)); }
        trans = env_at(dir, rb * mips) * tint;
    }
    let dielectric = spec_d + (1.0 - e_d) * ((1.0 - L.trans) * mix(diffuse, sss, L.sss) + L.trans * trans);
    var surface = mix(dielectric, spec_m, L.metal);
    if (L.coat > 0.0) {
        let cv = max(dot(fr.cn, v), 1e-4);
        let fc = coat_fresnel(L, 1.0);
        let ac = spec_albedo(vec3f(fc), vec3f(L.coat_scale), cv, L.coat_rough);
        let coat = env_at(reflect(-v, fr.cn), L.coat_rough * mips) * ac * specular_ao;
        surface = L.coat * coat + mix(vec3f(1.0), coat_transmittance(L, cv) * (1.0 - ac.x), L.coat)
                * darkening(L, cv) * surface;
    }
    if (L.fuzz > 0.0) {
        let ef = fuzz_albedo(nv, L.fuzz_rough);
        let fuzz = env_at(reflect(-v, n), max(L.fuzz_rough, 0.6) * mips) * L.fuzz_color * ef * ao;
        surface = L.fuzz * fuzz + (1.0 - L.fuzz * ef) * surface;
    }
    return surface;
}

@fragment
fn fs(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    gdx = dpdx(i.uv);
    gdy = dpdy(i.uv);
    var ng = normalize(i.normal);
    if (!front) { ng = -ng; }
    let t = normalize(i.tangent.xyz - ng * dot(ng, i.tangent.xyz));
    let bv = cross(ng, t) * i.tangent.w;  // along +V (down the image)
    let v = normalize(u.eye.xyz - i.world);
    var uv = i.uv;
    if (m.shader != UNLIT && m.shader != MESH_INFO) { uv = parallax(i.uv, vec3f(dot(v, t), dot(v, bv), dot(v, ng))); }
    let c = channels(uv);
    var alpha = 1.0;
    if (m.blending == ALPHA_BLEND) { alpha = clamp(c.opacity, 0.0, 1.0); }
    if (m.blending == ALPHA_TEST && c.opacity < m.alpha_threshold) { discard; }
    if (u.flags.z == 1u || m.shader == UNLIT) { return vec4f(display(c.base), alpha); }  // Preview / Unlit
    if (m.shader == MESH_INFO) {
        return vec4f(display(mesh_info(m.mode, ng, t, bv, i.mesh_uv, i.color, i.world)), 1.0);
    }
    let L = lobes(m, c);
    let ao = mix(1.0, L.ao, u.extra.x);
    var fr: Frame;
    fr.ng = ng;
    fr.n = mapped_normal(c.nrm, m.normal_scale, t, bv, ng);
    fr.cn = mapped_normal(c.coat_nrm, m.coat_normal_scale, t, bv, ng);
    let ca = cos(L.angle);
    let sa = sin(L.angle);
    let ta = t * ca - bv * sa;  // the tangent turned counterclockwise (Adobe Standard Material's angle)
    fr.t = normalize(ta - fr.n * dot(fr.n, ta));
    fr.b = cross(fr.n, fr.t);
    switch u.tone.y {
        case 1u: { return vec4f(display(c.base), 1.0); }
        case 2u: { return vec4f(display(fr.n * 0.5 + 0.5), 1.0); }
        case 3u: { return vec4f(display(vec3f(c.rough)), 1.0); }
        case 4u: { return vec4f(display(vec3f(c.metal)), 1.0); }
        case 5u: { return vec4f(display(vec3f(ao)), 1.0); }
        case 6u: { return vec4f(display(vec3f(c.height)), 1.0); }
        case 7u: { return vec4f(display(vec3f(fract(uv), 0.0)), 1.0); }
        default: {}
    }
    var color = vec3f(0.0);
    if (L.model != MODEL_LAMBERT) {  // Lambert: "the old point lights system", no image based light
        color += shade_light(L, fr, v, normalize(u.light_dir.xyz), u.light_color.rgb * u.light_dir.w)
               * mix(1.0, shadow(i.world, ng), u.extra.y);
        color += shade_ibl(L, fr, v, ao);
    } else {
        color += L.albedo * u.ambient.rgb * ao;
    }
    color += point_light(u.p0_pos, u.p0_col, i.world, L, fr, v);
    color += point_light(u.p1_pos, u.p1_col, i.world, L, fr, v);
    color += emitted(L, max(dot(fr.n, v), 1e-4));
    return vec4f(finish(color), alpha);
}

// The ground: a shadow catcher lit like a gray card, fading into the background with distance.
struct GOut { @builtin(position) pos: vec4f, @location(0) world: vec3f };

@vertex
fn ground_vs(@builtin(vertex_index) idx: u32) -> GOut {
    var c = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                            vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
    let w = vec3f(c[idx].x * 8.0, u.ground.w, c[idx].y * 8.0);
    var o: GOut;
    o.pos = u.view_proj * vec4f(w, 1.0);
    o.world = w;
    return o;
}

@fragment
fn ground_fs(i: GOut) -> @location(0) vec4f {
    let up = vec3f(0.0, 1.0, 0.0);
    let sun = u.light_color.rgb * u.light_dir.w * max(normalize(u.light_dir.xyz).y, 0.0);
    let lit = u.ground.rgb * (irradiance_at(up) + sun * mix(1.0, shadow(i.world, up), u.extra.z) / PI);
    let fade = 1.0 - smoothstep(1.8, 5.5, length(i.world.xz));
    return vec4f(finish(lit), fade);
}

struct BgOut { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };

@vertex
fn bg_vs(@builtin(vertex_index) idx: u32) -> BgOut {
    let p = vec2f(f32((idx << 1u) & 2u), f32(idx & 2u)) * 2.0 - 1.0;
    var o: BgOut;
    o.pos = vec4f(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

@fragment
fn bg_fs(i: BgOut) -> @location(0) vec4f {
    if (u.flags.y == 0u) { return vec4f(display(vec3f(0.137)), 1.0); }
    let far = u.inv_view_proj * vec4f(i.ndc, 1.0, 1.0);
    let near = u.inv_view_proj * vec4f(i.ndc, 0.0, 1.0);
    let d = normalize(far.xyz / far.w - near.xyz / near.w);
    let env = finish(env_at(d, u.env.z));
    if (u.bg_top.w >= 1.0) { return vec4f(env, 1.0); }
    let grad = display(mix(u.bg_bottom.rgb, u.bg_top.rgb, i.ndc.y * 0.5 + 0.5));
    return vec4f(mix(grad, env, u.bg_top.w), 1.0);
}

// Helper lines (grid, axis, bounding boxes, tangent frames, light helpers): world positions + color.
struct LOut { @builtin(position) pos: vec4f, @location(0) color: vec4f };
@vertex
fn lines_vs(@location(0) p: vec3f, @location(1) color: vec4f) -> LOut {
    var o: LOut;
    o.pos = u.view_proj * vec4f(p, 1.0);
    o.pos.z -= 1e-4 * o.pos.w;
    o.color = color;
    return o;
}
@fragment
fn lines_fs(i: LOut) -> @location(0) vec4f { return i.color; }
"""


AXIS_COLORS = ((1, 0.25, 0.25, 1), (0.3, 1, 0.3, 1), (0.35, 0.55, 1, 1))


def triangle_edges(idx) -> np.ndarray:
    """The wireframe's line list: every triangle's three sides. An edge two triangles share is drawn twice (it
    looks the same), which spares sorting millions of edges to find the unique ones (seconds on a tessellated
    mesh)."""
    return np.ascontiguousarray(idx.reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]].reshape(-1), np.uint32)


def _centre(points) -> np.ndarray:
    return (points.min(0) + points.max(0)) * 0.5 if len(points) else np.zeros(3, np.float32)


# The views' canvases draw at most this many frames a second (synced to the screen; the library's default, 30, made
# every view's navigation feel half as fluid on a 120 Hz screen, with a 3D frame at 5-8 ms).
MAX_FPS = 120.0


def present_method() -> str:
    """How the views' canvases show their frames: on the screen's native surface, or read back ("bitmap") when Qt
    runs offscreen (no native windows: creating a surface there crashes; the thumbnails tool, headless runs)."""
    from PySide6.QtGui import QGuiApplication
    return "bitmap" if QGuiApplication.platformName() == "offscreen" else "screen"


def world_axes_lines(view_proj, size) -> np.ndarray:
    """The world axes at the bottom right of the view, a fixed size in pixels (ours: 4.5% of the height, the
    center 1.2 lengths from the corner), turning with the camera: lines unprojected onto the near plane, so the
    helper-line shader puts them back on those pixels in front of everything."""
    w, h = size
    inv = np.linalg.inv(np.asarray(view_proj, np.float64))

    def world(ndc):
        p = inv @ np.array([ndc[0], ndc[1], ndc[2], 1.0])
        return p[:3] / p[3]
    o = world((0, 0, 0.5))
    right, up = world((0.01, 0, 0.5)) - o, world((0, 0.01, 0.5)) - o
    right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    length = 0.045 * h
    cx, cy = w - 2.2 * length, h - 2.2 * length  # pixels, y down
    out = []
    for axis, color in zip(np.eye(3), AXIS_COLORS):
        ex, ey = cx + float(np.dot(axis, right)) * length, cy - float(np.dot(axis, up)) * length
        for px, py in ((cx, cy), (ex, ey)):
            out.append((*world((px / w * 2 - 1, 1 - py / h * 2, 0.01)), *color))
    return np.array(out, np.float32)


def perspective(fov, aspect, near, far):
    f = 1 / math.tan(fov / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, far / (near - far), far * near / (near - far)], [0, 0, -1, 0]], np.float32)


def orthographic(half, near, far):
    return np.array([[1 / half, 0, 0, 0], [0, 1 / half, 0, 0],
                     [0, 0, -1 / (far - near), -near / (far - near)], [0, 0, 0, 1]], np.float32)


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.array(v, np.float32), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(float(np.dot(f, up))) > 0.999:
        up = np.array([0, 0, 1], np.float32)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def uv_to_dir(u, v):
    lon, lat = (u - 0.5) * 2 * math.pi, (0.5 - v) * math.pi
    return np.array([math.cos(lat) * math.sin(lon), math.sin(lat), -math.cos(lat) * math.cos(lon)], np.float32)


@dataclass
class Look:
    """What the 3D View's menus, toolbars and settings set (state files save it)."""
    env_rotation: float = 0.0
    env_intensity: float = 1.0
    exposure: float = 0.0  # stops
    tonemap: int = 2  # index in TONEMAPS: Linear, so base colors read as authored
    show_env: bool = True
    background_blur: float = 0.0  # mip level of the background
    env_opacity: float = 1.0  # the environment over the background gradient (Environment Opacity)
    background: tuple = ((0.3, 0.3, 0.3), (0.137, 0.137, 0.137))  # gradient top, bottom (sRGB; ours)
    world_axes: bool = False  # the world axes at the bottom right (Paint's General preferences)
    light_intensity: float = 1.0  # key light
    shadows: bool = True
    ground: bool = True
    displacement: float = 0.05  # Height scale: displacement of height 1 from the Height level
    height_level: float = 0.5
    enable_displacement: bool = True
    tessellation: int = 0  # subdivision levels of the meshes (for displacement)
    parallax: float = 0.0
    ao_opacity: float = 1.0
    shadow_opacity: float = 1.0
    ground_height: float = 0.0  # offset from the bottom of the mesh
    ground_shadow: float = 1.0
    backface_culling: bool = False
    # "Rasterizer order independent transparency enabled": Alpha blend as alpha to coverage, in any order
    # (faster, less accurate), else sorted blending. Ours off by default, the accurate one (Paint blends sorted
    # and has no such setting).
    order_independent_transparency: bool = False
    diagnostic: int = 0
    channel: int = 0  # Color channels: RGB, R, G, B
    opengl_normals: bool = False  # the project's normal format (default: DirectX)
    preview: bool = False
    # Display
    grid: bool = False
    axis: bool = False
    wireframe: bool = False
    bounding_box: bool = False
    tangent_space: bool = False
    light_helpers: bool = True
    # Renderer
    pathtracer: bool = False
    pt_samples: int = 256
    pt_bounces: int = 6
    ground_local_lights: bool = True
    ground_reflections: bool = True
    reflections_opacity: float = 0.3
    reflections_roughness: float = 0.15
    # Baking visualization (Paint's baking mode): the bake's high poly and cage over the low
    show_high: bool = False
    show_cage: bool = False
    show_rear: bool = False  # the rear distance surface (implicit cage only)
    cage_opacity: float = 0.3
    outside_red: bool = True  # high poly no bake ray reaches in red
    hide_low: bool = False


@dataclass
class GpuMesh:
    buffers: list
    index: object
    edges: object
    edge_count: int
    submeshes: list  # (start, count, material, centre of its bounding box: the blended surfaces' order)
    visible: bool
    bounds: tuple
    tangent_lines: np.ndarray


class Renderer:
    def __init__(self, device, color_format):
        from trafilatura3d.evaluator import COMPACT_FEATURE
        self.device = device
        self.compact = COMPACT_FEATURE in device.features  # one channel storage formats (r8unorm, r16float)
        self.format = color_format
        S, T, R = wgpu.TextureUsage.STORAGE_BINDING, wgpu.TextureUsage.TEXTURE_BINDING, wgpu.TextureUsage.RENDER_ATTACHMENT
        self.env = device.create_texture(size=(ENV_W, ENV_H, 1), format="rgba16float",
                                         usage=S | T | wgpu.TextureUsage.COPY_SRC, mip_level_count=ENV_MIPS)
        self.irradiance = device.create_texture(size=(IRR_W, IRR_H, 1), format="rgba16float", usage=S | T)
        self.env_small = device.create_texture(size=(IRR_SRC_W, IRR_SRC_H, 1), format="rgba16float", usage=S | T)
        self.brdf = device.create_texture(size=(LUT, LUT, 1), format="rgba16float", usage=S | T)
        self.shadow = device.create_texture(size=(SHADOW, SHADOW, 1), format="depth32float", usage=R | T)
        self.lin = device.create_sampler(mag_filter="linear", min_filter="linear", mipmap_filter="linear",
                                         address_mode_u="repeat", address_mode_v="clamp-to-edge")
        self.mat_sampler = device.create_sampler(mag_filter="linear", min_filter="linear", mipmap_filter="linear",
                                                 address_mode_u="repeat", address_mode_v="repeat",
                                                 max_anisotropy=16)
        self.cmp = device.create_sampler(mag_filter="linear", min_filter="linear", compare="less-equal")
        self.uniform = device.create_buffer(size=512, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._compute = {}
        self._constants = {}
        self._targets = {}
        self._pipelines = {}
        self.meshes: list[GpuMesh] = []
        self.materials: list = []  # per material: `material`'s dict
        self.selected: int | None = None
        self.bake_high: list = []  # baking visualization: (buffers, index, count) of the high poly meshes
        self.bake_cage: list = []  # (buffers, index, count, edges, edge count) of the cage
        self.bake_rear: list = []  # the same for the rear distance surface
        self.point_lights = []  # [(position, color, intensity)] of the enabled ones
        self.ambient = (0.0, 0.0, 0.0)
        self.last_draw_ms = 0.0
        self.extra_lines = None  # (N, 7) float32 lines drawn with the helpers
        # An image over the frame in screen space (the Paint Projection tool's preview): dict texture,
        # single_channel, xf (offset x, y, rotation, scale), tiling (0-3), opacity; None = none.
        self.screen_overlay = None
        self.hud = None  # `set_hud`: a client's image over the frame
        # The projection's falloff over the frame (Paint Projection tool): dict position / normal (the camera
        # buffer of the view), eye, fade (cos start, cos end), opacity; None = none.
        self.falloff_overlay = None
        self._build_layouts()
        enc = device.create_command_encoder()
        self._run(enc, "brdf", BRDF_LUT, [self.brdf.create_view()], (LUT, LUT))
        device.queue.submit([enc.finish()])
        self.set_studio()
        self.set_materials([{}])

    # --- pipelines --------------------------------------------------------------------------------

    def _build_layouts(self):
        d = self.device
        V, F = wgpu.ShaderStage.VERTEX, wgpu.ShaderStage.FRAGMENT
        self.bgl0 = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": V | F, "buffer": {"type": "uniform"}}])
        self.bgl1 = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": V | F, "buffer": {"type": "uniform"}}] + [
            {"binding": i, "visibility": V | F, "texture": {"sample_type": "float"}} for i in range(1, MAT_SAMPLER)] + [
            {"binding": MAT_SAMPLER, "visibility": V | F, "sampler": {"type": "filtering"}}])
        self.bgl2 = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 1, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 2, "visibility": F, "texture": {"sample_type": "float"}},
            {"binding": 3, "visibility": F, "sampler": {"type": "filtering"}},
            {"binding": 4, "visibility": F, "texture": {"sample_type": "depth"}},
            {"binding": 5, "visibility": F, "sampler": {"type": "comparison"}},
            {"binding": 6, "visibility": F, "texture": {"sample_type": "float", "view_dimension": "3d"}},
            {"binding": 7, "visibility": F, "sampler": {"type": "filtering"}}])
        self.module = d.create_shader_module(code=MAIN)
        self.group0 = d.create_bind_group(layout=self.bgl0, entries=[{"binding": 0, "resource": {"buffer": self.uniform}}])
        self.clamp_sampler = d.create_sampler(mag_filter="linear", min_filter="linear")
        self.ocio_lut = self._lut_texture(np.ones((1, 1, 1, 4), np.float32))
        self._make_group2()
        self._build_shadow_pipeline()

    def _lut_texture(self, lut):
        n = lut.shape[0]
        t = self.device.create_texture(size=(n, n, n), dimension="3d", format="rgba16float",
                                       usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": t}, np.ascontiguousarray(lut, np.float16).tobytes(),
                                        {"bytes_per_row": n * 8, "rows_per_image": n}, (n, n, n))
        return t

    def set_display_lut(self, lut):
        """OCIO view LUT (working space -> display / view, log2 shaped) for the "OCIO view" tone mapping."""
        self.ocio_lut = self._lut_texture(lut if lut is not None else np.ones((1, 1, 1, 4), np.float32))
        self._make_group2()

    def _make_group2(self):
        d = self.device
        self.group2 = d.create_bind_group(layout=self.bgl2, entries=[
            {"binding": 0, "resource": self.env.create_view()},
            {"binding": 1, "resource": self.irradiance.create_view()},
            {"binding": 2, "resource": self.brdf.create_view()},
            {"binding": 3, "resource": self.lin},
            {"binding": 4, "resource": self.shadow.create_view()},
            {"binding": 5, "resource": self.cmp},
            {"binding": 6, "resource": self.ocio_lut.create_view()},
            {"binding": 7, "resource": self.clamp_sampler}])

    def _build_shadow_pipeline(self):
        """The shadow map's pipelines: depth only (Opaque), with the opacity read (`shadow_fs`: the other modes)."""
        d = self.device
        layout = d.create_pipeline_layout(bind_group_layouts=[self.bgl0, self.bgl1])
        self.shadow_pipelines = [d.create_render_pipeline(
            layout=layout, vertex={"module": self.module, "entry_point": "shadow_vs", "buffers": VB},
            fragment={"module": self.module, "entry_point": "shadow_fs", "targets": []} if see_through else None,
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth32float", "depth_write_enabled": True, "depth_compare": "less",
                           "depth_bias": 2, "depth_bias_slope_scale": 2.0}) for see_through in (False, True)]

    def pipelines(self, fmt):
        """The render pipelines for a color format (the widget's, or rgba16float for HDR renders)."""
        if fmt in self._pipelines:
            return self._pipelines[fmt]
        d, mod = self.device, self.module
        full = d.create_pipeline_layout(bind_group_layouts=[self.bgl0, self.bgl1, self.bgl2])
        mesh_layout = d.create_pipeline_layout(bind_group_layouts=[self.bgl0, self.bgl1])
        only0 = d.create_pipeline_layout(bind_group_layouts=[self.bgl0])
        depth = {"format": "depth24plus", "depth_write_enabled": True, "depth_compare": "less"}
        test = {"format": "depth24plus", "depth_write_enabled": False, "depth_compare": "less-equal"}
        blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        target, blended = [{"format": fmt}], [{"format": fmt, "blend": blend}]

        def make(layout, vs, fs, targets, ds, topology="triangle-list", cull="none", buffers=VB, a2c=False):
            return d.create_render_pipeline(
                layout=layout, vertex={"module": mod, "entry_point": vs, "buffers": buffers},
                fragment={"module": mod, "entry_point": fs, "targets": targets},
                primitive={"topology": topology, "cull_mode": cull}, depth_stencil=ds,
                multisample={"count": SAMPLES, "alpha_to_coverage_enabled": a2c})
        behind = {"format": "depth24plus", "depth_write_enabled": False, "depth_compare": "less"}
        p = {
            "mesh": {(cull, a2c): make(full, "vs", "fs", target, depth, cull=cull, a2c=a2c)
                     for cull in ("none", "back") for a2c in (False, True)},
            "blend": {cull: make(full, "vs", "fs", blended, behind, cull=cull) for cull in ("back", "front")},
            "ground": make(full, "ground_vs", "ground_fs", blended, behind, buffers=[]),
            "bg": make(full, "bg_vs", "bg_fs", target,
                       {"format": "depth24plus", "depth_write_enabled": False, "depth_compare": "always"}, buffers=[]),
            "wire": make(mesh_layout, "wire_vs", "wire_fs", blended, test, topology="line-list"),
            "outline": make(mesh_layout, "outline_vs", "outline_fs", target, depth, cull="front"),
            "lines": make(only0, "lines_vs", "lines_fs", blended, test, topology="line-list", buffers=LINE_VB),
        }
        self._pipelines[fmt] = p
        return p

    def _run(self, enc, name, code, resources, size, *, buffers=()):
        """One compute dispatch: uniform `buffers` take the first bindings, then `resources`."""
        if name not in self._compute:
            self._compute[name] = self.device.create_compute_pipeline(
                layout="auto", compute={"module": self.device.create_shader_module(code=code), "entry_point": "main"})
        pipe = self._compute[name]
        entries = [{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(buffers)]
        entries += [{"binding": len(buffers) + i, "resource": r} for i, r in enumerate(resources)]
        group = self.device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=entries)
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((size[0] + 7) // 8, (size[1] + 7) // 8)
        cp.end()

    # --- environment ------------------------------------------------------------------------------

    def set_studio(self):
        enc = self.device.create_command_encoder()
        self._run(enc, "studio", STUDIO, [self._env_level(0)], (ENV_W, ENV_H))
        self._prefilter(enc)
        self._key(STUDIO_KEY, (1.0, 0.95, 0.88), 1.0)

    def set_environment(self, texture, light_dir, light_color=(1.0, 1.0, 1.0), light_strength=2.0):
        """A float lat-long texture (any size and float format) as the environment; the key light
        (and its shadow) comes from `light_dir`, a direction of the panorama."""
        enc = self.device.create_command_encoder()
        self._run(enc, "resample", RESAMPLE, [texture.create_view(), self._env_level(0)], (ENV_W, ENV_H))
        self._prefilter(enc)
        self._key(light_dir, light_color, light_strength)

    def _key(self, direction, color, strength):
        d = np.asarray(direction, np.float32)
        self.light_dir = d / max(float(np.linalg.norm(d)), 1e-6)
        self.light_color = np.asarray(color, np.float32)
        self.light_strength = strength

    def set_environment_image(self, pixels: np.ndarray):
        """An HDR panorama as (height, width, channels) float32: uploaded, key light at its
        brightest region."""
        h, w = pixels.shape[:2]
        rgba = np.ones((h, w, 4), np.float32)
        if pixels.shape[2] < 3:
            rgba[..., :3] = pixels[..., :1]
        else:
            rgba[..., :3] = pixels[..., :3]
        t = self.device.create_texture(size=(w, h, 1), format="rgba32float",
                                       usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": t}, np.ascontiguousarray(rgba).tobytes(),
                                        {"bytes_per_row": w * 16}, (w, h, 1))
        # Brightest region on a coarse grid (block means), its color normalized as the key's tint.
        bh, bw = max(h // 32, 1), max(w // 64, 1)
        gh, gw = h // bh, w // bw
        coarse = rgba[:gh * bh, :gw * bw, :3].reshape(gh, bh, gw, bw, 3).mean((1, 3))
        lum = coarse @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        y, x = np.unravel_index(int(np.argmax(lum)), lum.shape)
        color = coarse[y, x] / max(float(coarse[y, x].max()), 1e-6)
        self.set_environment(t, uv_to_dir((x + 0.5) / gw, (y + 0.5) / gh), color)

    def _env_level(self, level):
        return self.env.create_view(base_mip_level=level, mip_level_count=1)

    def _prefilter(self, enc):
        prev = 0.0
        for k in range(1, ENV_MIPS):
            alpha = (k / (ENV_MIPS - 1)) ** 2
            inc = math.sqrt(max(alpha * alpha - prev * prev, 1e-8))
            prev = alpha
            buf = self.device.create_buffer_with_data(data=struct.pack("4f", inc, 0, 0, 0), usage=wgpu.BufferUsage.UNIFORM)
            self._run(enc, "prefilter", PREFILTER, [self._env_level(k - 1), self.lin, self._env_level(k)],
                      (ENV_W >> k, ENV_H >> k), buffers=[buf])
        self._run(enc, "reduce", RESAMPLE, [self._env_level(0), self.env_small.create_view()], (IRR_SRC_W, IRR_SRC_H))
        self._run(enc, "irradiance", IRRADIANCE, [self.env_small.create_view(), self.irradiance.create_view()],
                  (IRR_W, IRR_H))
        self.device.queue.submit([enc.finish()])

    # --- meshes and materials ---------------------------------------------------------------------

    def set_meshes(self, objects):
        """Scene objects (scene3d.MeshObject) to GPU buffers, with edge lists for the wireframe and
        a few tangent frames for the vertex tangent space display."""
        mk = lambda a, usage: self.device.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(), usage=usage)
        V, I = wgpu.BufferUsage.VERTEX, wgpu.BufferUsage.INDEX
        self.meshes = []
        for o in objects:
            edges = triangle_edges(o.idx)
            step = max(1, len(o.pos) // 1500)
            p, n, t = o.pos[::step], o.nrm[::step], o.tan[::step, :3]
            b = np.cross(n, t) * o.tan[::step, 3:]
            L = 0.06
            lines = []
            for d, c in ((t, (1, 0.2, 0.2, 1)), (b, (0.2, 1, 0.2, 1)), (n, (0.3, 0.5, 1, 1))):
                seg = np.empty((len(p) * 2, 7), np.float32)
                seg[0::2, :3], seg[1::2, :3] = p, p + d * L
                seg[:, 3:] = c
                lines.append(seg)
            colors = getattr(o, "colors", None)  # Mesh Info's vertex color: white without
            colors = np.ones((len(o.pos), 4), np.float32) if colors is None else np.asarray(colors, np.float32)
            self.meshes.append(GpuMesh([mk(o.pos, V), mk(o.nrm, V), mk(o.tan, V), mk(o.uv, V), mk(colors, V)], mk(o.idx, I),
                                       mk(edges, I), edges.size,
                                       [(s.start, s.count, s.material, _centre(o.pos[o.idx[s.start:s.start + s.count]]))
                                        for s in o.submeshes], o.visible,
                                       o.bounds(), np.concatenate(lines)))

    def set_visibility(self, visible):
        for m, v in zip(self.meshes, visible):
            m.visible = v

    def set_materials(self, materials: list):
        """Per scene material a dict: slot -> (texture, is_gray) for the bound channels, plus "shader" (index in
        shaders.SHADERS), "uniforms" ({field: value}, `material_uniform`), "constants" {slot: rgba} for unbound
        channels. Textures are copied into mipmapped textures (`material_format`)."""
        self.materials = [self.material(spec) for spec in materials]

    def material(self, spec: dict, kept: dict | None = None) -> dict:
        """One material of `set_materials` (its textures copied now: the sources can go once this returns). `kept`:
        slot -> a mipmapped copy this renderer already has, taken as it is (its source may be gone)."""
        enc = self.device.create_command_encoder()
        usage = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
        textures = {}
        for slot in SLOTS:
            if kept and slot in kept:
                textures[slot] = kept[slot]
                continue
            if slot not in spec:
                textures[slot] = self._constant(spec.get("constants", {}).get(slot, DEFAULTS[slot]))
                continue
            src, gray = spec[slot]
            w, h = src.size[0], src.size[1]
            mips = int(math.log2(max(w, h))) + 1
            fmt = self.material_format(slot, src)
            t = self.device.create_texture(size=(w, h, 1), format=fmt, usage=usage, mip_level_count=mips)
            self._convert(enc, src, gray, t)
            textures[slot] = t
        ubuf = self.device.create_buffer_with_data(data=material_uniform(spec), usage=wgpu.BufferUsage.UNIFORM)
        group = self.device.create_bind_group(layout=self.bgl1, entries=[
            {"binding": 0, "resource": {"buffer": ubuf}}] + [
            {"binding": i + 1, "resource": textures[s].create_view()} for i, s in enumerate(SLOTS)] + [
            {"binding": MAT_SAMPLER, "resource": self.mat_sampler}])
        self.device.queue.submit([enc.finish()])
        return {"group": group, "blending": blending(spec), "ubuf": ubuf, "spec": spec, "textures": textures}

    def update_slots(self, index: int, slots: dict, rect=None) -> None:
        """Refresh some slots of one material ({slot: (texture, is_gray)}) in place: the mipmapped copies are
        rewritten when their size is unchanged (no new textures or bind group; the Paint brush calls this
        while painting), else the material is rebuilt. `rect` (x0, y0, x1, y1): only the texels there changed
        (the brush's), and the mips over them."""
        mat = self.materials[index]
        spec = {**mat["spec"], **slots}
        same = all(slot in mat["spec"] and tuple(mat["textures"][slot].size[:2]) == tuple(src.size[:2])
                   and mat["textures"][slot].format == self.material_format(slot, src)
                   for slot, (src, _g) in slots.items())
        if not same:  # the other slots' copies kept (their sources may be gone)
            kept = {slot: t for slot, t in mat["textures"].items() if slot in mat["spec"] and slot not in slots}
            self.materials[index] = self.material(spec, kept)
            return
        enc = self.device.create_command_encoder()
        for slot, (src, gray) in slots.items():
            self._convert(enc, src, gray, mat["textures"][slot], rect)
        self.device.queue.submit([enc.finish()])
        mat["spec"] = spec

    # Kept for the single-material callers (tests): one material on everything.
    def set_textures(self, sources: dict):
        self.set_materials([sources])

    def set_mesh(self, pos, nrm, tan, uv, idx):
        from trafilatura3d.editor.scene3d import MeshObject, Submesh
        self.set_meshes([MeshObject("mesh", pos, nrm, tan, uv, idx, [Submesh(0, idx.size, 0)])])

    def _constant(self, value):
        key = tuple(round(float(x), 5) for x in value)
        if key not in self._constants:  # read back like any slot (`read_material`: the path tracer's flat height)
            t = self.device.create_texture(size=(1, 1, 1), format="rgba16float",
                                           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
                                           | wgpu.TextureUsage.COPY_SRC)
            self.device.queue.write_texture({"texture": t}, np.array(key, np.float16).tobytes(),
                                            {"bytes_per_row": 8}, (1, 1, 1))
            self._constants[key] = t
        return self._constants[key]

    def material_format(self, slot: str, src) -> str:
        """The mipmapped copy's format: one channel for the slots read as one value, 8 bits from an 8 bit source
        (a 4096 roughness: 21 MB with its mips instead of 170 in half float RGBA)."""
        eight = src.format in EIGHT_BIT
        if slot in SINGLE and self.compact:
            return "r8unorm" if eight else "r16float"
        return "rgba8unorm" if eight else "rgba16float"

    def _convert(self, enc, src, gray: bool, t, rect=None) -> None:
        """`src` into the material texture `t`: level 0 converted (gray spread to RGB), then each mip the 2x2 means;
        `rect` (x0, y0, x1, y1): only that rectangle of level 0 and the texels of each mip over it."""
        w, h = t.size[0], t.size[1]
        x0, y0, x1, y1 = rect or (0, 0, w, h)
        level = lambda k: t.create_view(base_mip_level=k, mip_level_count=1)
        fmt = t.format
        uni = lambda *v: self.device.create_buffer_with_data(data=struct.pack("4I", *v, *(0,) * (4 - len(v))),
                                                             usage=wgpu.BufferUsage.UNIFORM)
        self._run(enc, f"to_half:{fmt}", TO_HALF.replace("FORMAT", fmt), [src.create_view(), level(0)],
                  (x1 - x0, y1 - y0), buffers=[uni(int(gray), x0, y0)])
        for k in range(1, t.mip_level_count):
            x0, y0, x1, y1 = x0 >> 1, y0 >> 1, max((x1 + 1) >> 1, (x0 >> 1) + 1), max((y1 + 1) >> 1, (y0 >> 1) + 1)
            x1, y1 = min(x1, max(w >> k, 1)), min(y1, max(h >> k, 1))
            self._run(enc, f"downsample:{fmt}", DOWNSAMPLE.replace("FORMAT", fmt), [level(k - 1), level(k)],
                      (x1 - x0, y1 - y0), buffers=[uni(x0, y0)])

    def material_texture(self, index, slot):
        """The texture a material shows in a slot (`material_format`)."""
        return self.materials[min(index, len(self.materials) - 1)]["textures"][slot]

    def read_material(self, index, slot) -> np.ndarray:
        """A material slot's level 0 as float32 HxWx4 (a one channel slot spread to RGB, alpha 1): exports and the
        path tracer's displacement."""
        t = self.material_texture(index, slot)
        w, h = t.size[0], t.size[1]
        dtype, channels = {"r8unorm": (np.uint8, 1), "r16float": (np.float16, 1), "rgba8unorm": (np.uint8, 4),
                           "rgba16float": (np.float16, 4)}[t.format]
        row = (w * channels * np.dtype(dtype).itemsize + 255) // 256 * 256
        data = self.device.queue.read_texture({"texture": t}, {"bytes_per_row": row}, (w, h, 1))
        a = np.frombuffer(data, np.uint8).reshape(h, row)[:, :w * channels * np.dtype(dtype).itemsize]
        a = a.copy().view(dtype).reshape(h, w, channels).astype(np.float32)
        if dtype == np.uint8:
            a /= 255.0
        if channels == 1:
            a = np.concatenate([a, a, a, np.ones_like(a)], 2)
        return a

    # --- frame ------------------------------------------------------------------------------------

    def world_light(self, look: Look):
        """The key light in world space: the panorama's key direction turned with the environment
        (env_at samples rot_y(d), so a panorama direction shows up at rot_y^-1 of it)."""
        a = -look.env_rotation
        x, y, z = self.light_dir
        return np.array([x * math.cos(a) - z * math.sin(a), y, x * math.sin(a) + z * math.cos(a)], np.float32)

    def scene_bounds(self):
        vis = [m for m in self.meshes if m.visible] or self.meshes
        return np.min([m.bounds[0] for m in vis], 0), np.max([m.bounds[1] for m in vis], 0)

    def _overlay(self, look: Look, view_proj, size):
        """Helper lines for this frame: (N, 7) float32 position + color pairs (plus `extra_lines`, set by a
        client such as the Paint brush cursor)."""
        parts = [self.extra_lines] if self.extra_lines is not None and len(self.extra_lines) else []
        if look.world_axes:
            parts.append(world_axes_lines(view_proj, size))
        lo, hi = self.scene_bounds()
        height = look.displacement if look.enable_displacement else 0.0
        floor = lo[1] - abs(height) * look.height_level + look.ground_height
        if look.grid:
            r = np.arange(-5, 6, 0.5, dtype=np.float32)
            g = []
            for x in r:
                c = (0.55, 0.55, 0.55, 0.7) if x == 0 else (0.4, 0.4, 0.4, 0.35)
                g += [(x, floor, -5, *c), (x, floor, 5, *c), (-5, floor, x, *c), (5, floor, x, *c)]
            parts.append(np.array(g, np.float32))
        if look.axis:
            a = []
            for d, c in (((1, 0, 0), (1, 0.25, 0.25, 1)), ((0, 1, 0), (0.3, 1, 0.3, 1)), ((0, 0, 1), (0.35, 0.55, 1, 1))):
                a += [(0, 0, 0, *c), (*(np.array(d) * 1.2), *c)]
            parts.append(np.array(a, np.float32))
        if look.bounding_box:
            for m in self.meshes:
                if not m.visible:
                    continue
                b0, b1 = m.bounds
                corners = np.array([[x, y, z] for x in (b0[0], b1[0]) for y in (b0[1], b1[1]) for z in (b0[2], b1[2])])
                e = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
                seg = np.array([[*corners[k], 1.0, 0.85, 0.2, 1.0] for pair in e for k in pair], np.float32)
                parts.append(seg)
        if look.tangent_space:
            parts += [m.tangent_lines for m in self.meshes if m.visible]
        if look.light_helpers:
            for pos, color, _ in self.point_lights:
                p = np.asarray(pos, np.float32)
                c = (*color, 1.0)
                s = 0.08
                parts.append(np.array([(*p, *c), (0, 0, 0, *c[:3], 0.3)] + [
                    v for d in np.eye(3) for v in ((*(p - d * s), *c), (*(p + d * s), *c))], np.float32))
        return np.concatenate(parts) if parts else None

    def draw(self, target, eye, view_proj, look: Look, linear=False):
        """One frame into `target` (a texture view's texture). `linear` writes scene linear values
        with exposure and no tone mapping (HDR renders into rgba16float)."""
        import time
        t0 = time.perf_counter()
        fmt = target.format
        w, h = target.size[0], target.size[1]
        key = (fmt, w, h)
        if key not in self._targets:
            R = wgpu.TextureUsage.RENDER_ATTACHMENT
            self._targets = {key: (
                self.device.create_texture(size=(w, h, 1), format=fmt, usage=R, sample_count=SAMPLES),
                self.device.create_texture(size=(w, h, 1), format="depth24plus", usage=R, sample_count=SAMPLES)),
                **{k: v for k, v in self._targets.items() if k[1:] != (w, h) or k[0] != fmt}}
        color_t, depth_t = self._targets[key]
        pipes = self.pipelines(fmt)
        light = self.world_light(look)
        height = look.displacement if look.enable_displacement else 0.0
        lo, hi = self.scene_bounds()
        ground_y = float(lo[1]) - abs(height) * look.height_level - 0.003 + look.ground_height
        light_vp = orthographic(SHADOW_EXTENT, 0.1, 20.0) @ look_at(light * 8.0, (0, 0, 0))
        inv_vp = np.linalg.inv(view_proj)
        pl = list(self.point_lights)[:2] + [None] * (2 - len(self.point_lights[:2]))
        lights = []
        for p in pl:
            lights += [*p[0], 1.0, *p[1], p[2]] if p else [0.0] * 8
        data = b"".join(m.T.astype(np.float32).tobytes() for m in (view_proj, light_vp, inv_vp)) + struct.pack(
            "4f 4f 4f 4f 4f 4I 4f 4I 4f 4f 8f 8f 4f 4f 4f",
            *eye, 1.0,
            *light, look.light_intensity * self.light_strength,
            *self.light_color, 1.0,
            height, 0.004, look.exposure, look.parallax,
            look.env_rotation, look.env_intensity, look.background_blur, float(ENV_MIPS - 1),
            int(look.opengl_normals), int(look.show_env), int(look.preview), int(look.shadows),
            0.18, 0.18, 0.18, ground_y,
            look.tonemap, look.diagnostic, look.channel, int(linear),
            look.ao_opacity, look.shadow_opacity, look.ground_shadow, look.height_level,
            *self.ambient, 0.0,
            *lights[:8], *lights[8:],
            0.85, 0.85, 0.85, 0.55,
            *look.background[0], look.env_opacity, *look.background[1], 0.0)
        self.device.queue.write_buffer(self.uniform, 0, data)
        visible = [] if look.hide_low and self.bake_high else [m for m in self.meshes if m.visible]
        enc = self.device.create_command_encoder()
        if look.shadows and not look.preview:
            sp = enc.begin_render_pass(color_attachments=[], depth_stencil_attachment={
                "view": self.shadow.create_view(), "depth_load_op": "clear", "depth_store_op": "store",
                "depth_clear_value": 1.0})
            sp.set_bind_group(0, self.group0)
            for m in visible:
                self._submeshes(sp, m, lambda mat: self.shadow_pipelines[mat["blending"] != shaders.OPAQUE])
            sp.end()
        rp = enc.begin_render_pass(
            color_attachments=[{"view": color_t.create_view(), "resolve_target": target.create_view(),
                                "load_op": "clear", "store_op": "discard", "clear_value": (0.137, 0.137, 0.137, 1)}],
            depth_stencil_attachment={"view": depth_t.create_view(), "depth_load_op": "clear",
                                      "depth_store_op": "discard", "depth_clear_value": 1.0})
        rp.set_bind_group(0, self.group0)
        rp.set_bind_group(2, self.group2)
        rp.set_bind_group(1, self.materials[0]["group"])
        rp.set_pipeline(pipes["bg"])
        rp.draw(3)
        if look.ground and not look.preview:
            rp.set_pipeline(pipes["ground"])
            rp.draw(6)
        cull = "back" if look.backface_culling else "none"
        oit = look.order_independent_transparency
        sorted_blend = lambda mat: mat["blending"] == shaders.ALPHA_BLEND and not oit
        for m in visible:
            self._submeshes(rp, m, lambda mat: None if sorted_blend(mat) else
                            pipes["mesh"][(cull, mat["blending"] == shaders.ALPHA_BLEND)])
        self._draw_blended(rp, pipes, visible, eye, look, sorted_blend)
        if self.selected is not None and self.selected < len(self.meshes) and self.meshes[self.selected].visible:
            self._submeshes(rp, self.meshes[self.selected], lambda mat: pipes["outline"])
        if look.wireframe:
            rp.set_pipeline(pipes["wire"])
            for m in visible:
                self._vertex_buffers(rp, m)
                rp.set_bind_group(1, self.materials[min(m.submeshes[0][2], len(self.materials) - 1)]["group"])
                rp.set_index_buffer(m.edges, "uint32")
                rp.draw_indexed(m.edge_count)
        self._draw_bake(rp, fmt, look)
        lines = self._overlay(look, view_proj, (w, h))
        if lines is not None and len(lines):
            buf = self.device.create_buffer_with_data(data=lines.tobytes(), usage=wgpu.BufferUsage.VERTEX)
            rp.set_pipeline(pipes["lines"])
            rp.set_vertex_buffer(0, buf)
            rp.draw(len(lines))
        if self.falloff_overlay is not None:
            self._draw_falloff(rp, fmt, (w, h))
        if self.screen_overlay is not None:
            self._draw_screen_overlay(rp, fmt, w / max(h, 1))
        if self.hud is not None:
            self._draw_hud(rp, fmt, (w, h))
        rp.end()
        self.device.queue.submit([enc.finish()])
        self.last_draw_ms = (time.perf_counter() - t0) * 1000

    def _draw_falloff(self, rp, fmt, size):
        """`falloff_overlay`: red where the projection fades or does not reach (projection_weight)."""
        o = self.falloff_overlay
        d = self.device
        key = ("falloff_overlay", fmt)
        if key not in self._pipelines:
            from trafilatura3d.paint.brush import PROJECTION_WGSL
            mod = d.create_shader_module(code=PROJECTION_WGSL + FALLOFF_OVERLAY)
            blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                     "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
            self._pipelines[key] = d.create_render_pipeline(
                layout="auto", vertex={"module": mod, "entry_point": "vs", "buffers": []},
                fragment={"module": mod, "entry_point": "fs", "targets": [{"format": fmt, "blend": blend}]},
                primitive={"topology": "triangle-list"},
                depth_stencil={"format": "depth24plus", "depth_write_enabled": False, "depth_compare": "always"},
                multisample={"count": SAMPLES})
        pipe = self._pipelines[key]
        uni = struct.pack("12f", *o["eye"], float(o["opacity"]), *o["fade"], 0.0, 0.0, float(size[0]), float(size[1]),
                          0.0, 0.0)
        group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": o["position"].create_view()},
            {"binding": 2, "resource": o["normal"].create_view()}])
        rp.set_pipeline(pipe)
        rp.set_bind_group(0, group)
        rp.draw(3)

    def set_hud(self, image: np.ndarray | None, x: int = 0, y: int = 0) -> None:
        """An RGBA8 image (premultiplied, HxWx4 uint8) drawn over the frame with its top left at pixel (x, y), or
        None: text helpers of a client (Paint's keyboard helper)."""
        if image is None:
            self.hud = None
            return
        h, w = image.shape[:2]
        t = self.device.create_texture(size=(w, h, 1), format="rgba8unorm",
                                       usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": t}, np.ascontiguousarray(image).tobytes(),
                                        {"bytes_per_row": w * 4}, (w, h, 1))
        self.hud = {"texture": t, "rect": (float(x), float(y), float(w), float(h))}

    def _draw_hud(self, rp, fmt, size):
        d = self.device
        key = ("hud", fmt)
        if key not in self._pipelines:
            mod = d.create_shader_module(code=HUD)
            blend = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                     "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
            self._pipelines[key] = d.create_render_pipeline(
                layout="auto", vertex={"module": mod, "entry_point": "vs", "buffers": []},
                fragment={"module": mod, "entry_point": "fs", "targets": [{"format": fmt, "blend": blend}]},
                primitive={"topology": "triangle-list"},
                depth_stencil={"format": "depth24plus", "depth_write_enabled": False, "depth_compare": "always"},
                multisample={"count": SAMPLES})
        pipe = self._pipelines[key]
        uni = struct.pack("8f", *self.hud["rect"], float(size[0]), float(size[1]), 0.0, 0.0)
        group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": self.hud["texture"].create_view()}])
        rp.set_pipeline(pipe)
        rp.set_bind_group(0, group)
        rp.draw(6)

    def _draw_screen_overlay(self, rp, fmt, aspect):
        """`screen_overlay` over the frame, through the projection of paint/wgsl/projection.wgsl."""
        o = self.screen_overlay
        d = self.device
        key = ("screen_overlay", fmt)
        if key not in self._pipelines:
            from trafilatura3d.paint.brush import PROJECTION_WGSL
            code = PROJECTION_WGSL + SCREEN_OVERLAY
            mod = d.create_shader_module(code=code)
            blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                     "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
            self._pipelines[key] = d.create_render_pipeline(
                layout="auto", vertex={"module": mod, "entry_point": "vs", "buffers": []},
                fragment={"module": mod, "entry_point": "fs", "targets": [{"format": fmt, "blend": blend}]},
                primitive={"topology": "triangle-list"},
                depth_stencil={"format": "depth24plus", "depth_write_enabled": False, "depth_compare": "always"},
                multisample={"count": SAMPLES})
        pipe = self._pipelines[key]
        uni = struct.pack("8f", *o["xf"], aspect, float(o["tiling"]), float(o["opacity"]),
                          1.0 if o.get("single_channel") else 0.0)  # tiling: with the flips (projection.wgsl)
        group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": o["texture"].create_view()}])
        rp.set_pipeline(pipe)
        rp.set_bind_group(0, group)
        rp.draw(3)

    # --- baking visualization ------------------------------------------------------------------------

    def set_bake_overlay(self, high: list, outside: list, cage: list, rear: list):
        """The bake's high poly meshes (with a per vertex "no ray reaches it" flag), cage and rear distance
        surface, in view space."""
        mk = lambda a, usage: self.device.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(), usage=usage)
        V, I = wgpu.BufferUsage.VERTEX, wgpu.BufferUsage.INDEX
        self.bake_high = [([mk(o.pos, V), mk(o.nrm, V), mk(np.asarray(f, np.float32), V)], mk(o.idx, I), o.idx.size)
                          for o, f in zip(high, outside)]

        def shell(objects):
            out = []
            for o in objects:
                edges = triangle_edges(o.idx)
                zero = np.zeros(len(o.pos), np.float32)
                out.append(([mk(o.pos, V), mk(o.nrm, V), mk(zero, V)], mk(o.idx, I), o.idx.size,
                            mk(edges, I), edges.size))
            return out
        self.bake_cage, self.bake_rear = shell(cage), shell(rear)

    def _bake_pipelines(self, fmt):
        key = ("bake", fmt)
        if key not in self._pipelines:
            d = self.device
            mod = d.create_shader_module(code=BAKE_VIS)
            self._bake_bgl = getattr(self, "_bake_bgl", None) or d.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": "uniform"}}])
            layout = d.create_pipeline_layout(bind_group_layouts=[self.bgl0, self._bake_bgl])
            blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                     "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
            buffers = [{"array_stride": 12, "attributes": [{"shader_location": 0, "offset": 0, "format": "float32x3"}]},
                       {"array_stride": 12, "attributes": [{"shader_location": 1, "offset": 0, "format": "float32x3"}]},
                       {"array_stride": 4, "attributes": [{"shader_location": 2, "offset": 0, "format": "float32"}]}]

            def make(topology, write, blended):
                return d.create_render_pipeline(
                    layout=layout, vertex={"module": mod, "entry_point": "vs", "buffers": buffers},
                    fragment={"module": mod, "entry_point": "fs",
                              "targets": [{"format": fmt, **({"blend": blend} if blended else {})}]},
                    primitive={"topology": topology, "cull_mode": "none"},
                    depth_stencil={"format": "depth24plus", "depth_write_enabled": write,
                                   "depth_compare": "less" if write else "less-equal"},
                    multisample={"count": SAMPLES})
            self._pipelines[key] = {"solid": make("triangle-list", True, False),
                                    "glass": make("triangle-list", False, True),
                                    "edges": make("line-list", False, True)}
        return self._pipelines[key]

    def _bake_group(self, color, mode):
        buf = self.device.create_buffer_with_data(data=struct.pack("4f 4f", *color, float(mode), 0, 0, 0),
                                                  usage=wgpu.BufferUsage.UNIFORM)
        return self.device.create_bind_group(layout=self._bake_bgl, entries=[{"binding": 0, "resource": {"buffer": buf}}])

    def _draw_bake(self, rp, fmt, look: Look):
        if not ((look.show_high and self.bake_high) or (look.show_cage and self.bake_cage)
                or (look.show_rear and self.bake_rear)):
            return
        pipes = self._bake_pipelines(fmt)
        rp.set_bind_group(0, self.group0)
        if look.show_high:
            rp.set_pipeline(pipes["solid"])
            rp.set_bind_group(1, self._bake_group((0.62, 0.62, 0.64, 1.0), 1.0 if look.outside_red else 0.0))
            for bufs, index, count in self.bake_high:
                for i, b in enumerate(bufs):
                    rp.set_vertex_buffer(i, b)
                rp.set_index_buffer(index, "uint32")
                rp.draw_indexed(count)
        # Glass: the rear distance surface in orange under the cage in blue, edges on top.
        for on, shells, color in ((look.show_rear, self.bake_rear, (1.0, 0.55, 0.15)),
                                  (look.show_cage, self.bake_cage, (0.25, 0.55, 1.0))):
            if not on or not shells:
                continue
            rp.set_pipeline(pipes["glass"])
            rp.set_bind_group(1, self._bake_group((*color, look.cage_opacity), 2.0))
            for bufs, index, count, _, _ in shells:
                for i, b in enumerate(bufs):
                    rp.set_vertex_buffer(i, b)
                rp.set_index_buffer(index, "uint32")
                rp.draw_indexed(count)
            rp.set_pipeline(pipes["edges"])
            edge = tuple(min(1.0, c + 0.2) for c in color)
            rp.set_bind_group(1, self._bake_group((*edge, min(1.0, look.cage_opacity + 0.35)), 3.0))
            for bufs, _, _, edges, edge_count in shells:
                for i, b in enumerate(bufs):
                    rp.set_vertex_buffer(i, b)
                rp.set_index_buffer(edges, "uint32")
                rp.draw_indexed(edge_count)

    def _draw_blended(self, rp, pipes, meshes, eye, look, blended):
        """The sorted Alpha blend submeshes after the opaque ones: farthest bounding box centre first, each its back
        faces then its front faces (front only with backface culling)."""
        order = []
        for m in meshes:
            for start, count, mi, centre in m.submeshes:
                mat = self.materials[min(mi, len(self.materials) - 1)]
                if blended(mat):
                    order.append((float(np.linalg.norm(np.asarray(centre) - np.asarray(eye[:3]))), m, start, count, mat))
        order.sort(key=lambda o: -o[0])
        faces = ("back",) if look.backface_culling else ("front", "back")  # the culled side
        for _, m, start, count, mat in order:
            self._vertex_buffers(rp, m)
            rp.set_index_buffer(m.index, "uint32")
            rp.set_bind_group(1, mat["group"])
            for cull in faces:
                rp.set_pipeline(pipes["blend"][cull])
                rp.draw_indexed(count, 1, start)

    def _vertex_buffers(self, rp, m):
        for i, b in enumerate(m.buffers):
            rp.set_vertex_buffer(i, b)

    def _submeshes(self, rp, m, pipeline_of):
        """A mesh's submeshes, each with the pipeline `pipeline_of` gives its material (None: not drawn here)."""
        self._vertex_buffers(rp, m)
        rp.set_index_buffer(m.index, "uint32")
        for start, count, mi, _ in m.submeshes:
            mat = self.materials[min(mi, len(self.materials) - 1)]
            pipeline = pipeline_of(mat)
            if pipeline is None:
                continue
            rp.set_pipeline(pipeline)
            rp.set_bind_group(1, mat["group"])
            rp.draw_indexed(count, 1, start)


VB = [{"array_stride": 12, "attributes": [{"shader_location": 0, "offset": 0, "format": "float32x3"}]},
      {"array_stride": 12, "attributes": [{"shader_location": 1, "offset": 0, "format": "float32x3"}]},
      {"array_stride": 16, "attributes": [{"shader_location": 2, "offset": 0, "format": "float32x4"}]},
      {"array_stride": 8, "attributes": [{"shader_location": 3, "offset": 0, "format": "float32x2"}]},
      {"array_stride": 16, "attributes": [{"shader_location": 4, "offset": 0, "format": "float32x4"}]}]
LINE_VB = [{"array_stride": 28, "attributes": [{"shader_location": 0, "offset": 0, "format": "float32x3"},
                                               {"shader_location": 1, "offset": 12, "format": "float32x4"}]}]

# Baking visualization: the high poly as a neutral clay lit from the camera (red where it lies outside the
# cage, mode 1), the cage as tinted glass (mode 2) and its edges (mode 3). Reads the frame's uniform (U).
SCREEN_OVERLAY = """
struct OverlayU { xf: vec4f, params: vec4f };  // offset, rotation, scale; aspect, tiling, opacity, single channel
@group(0) @binding(0) var<uniform> ou: OverlayU;
@group(0) @binding(1) var o_tex: texture_2d<f32>;
struct OV { @builtin(position) pos: vec4f, @location(0) s: vec2f };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> OV {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return OV(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
}
@fragment
fn fs(i: OV) -> @location(0) vec4f {
    let q = projection_uv(i.s, ou.params.x, ou.xf, i32(ou.params.y));
    var t = bilinear_wrap(o_tex, q.xy);
    if (ou.params.w > 0.5) { t = vec4f(t.rrr, 1.0); }
    return vec4f(t.rgb, t.a * q.z * ou.params.z);
}
"""

HUD = """
struct HudU { rect: vec4f, frame: vec4f };  // x, y, width, height in pixels (y down); frame width, height
@group(0) @binding(0) var<uniform> hu: HudU;
@group(0) @binding(1) var h_tex: texture_2d<f32>;
struct HV { @builtin(position) pos: vec4f };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> HV {
    let corners = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1), vec2f(0, 1));
    let px = hu.rect.xy + corners[i] * hu.rect.zw;
    return HV(vec4f(px.x / hu.frame.x * 2.0 - 1.0, 1.0 - px.y / hu.frame.y * 2.0, 0.0, 1.0));
}
@fragment
fn fs(i: HV) -> @location(0) vec4f {
    let t = textureLoad(h_tex, vec2i(i.pos.xy - hu.rect.xy), 0);
    return t;
}
"""

FALLOFF_OVERLAY = """
struct FalloffU { eye: vec4f, fade: vec4f, frame: vec4f };  // eye + opacity; cos start, cos end; frame size
@group(0) @binding(0) var<uniform> fu: FalloffU;
@group(0) @binding(1) var f_pos: texture_2d<f32>;
@group(0) @binding(2) var f_nrm: texture_2d<f32>;
struct FV { @builtin(position) pos: vec4f };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> FV {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return FV(vec4f(p * 2.0 - 1.0, 0.0, 1.0));
}
@fragment
fn fs(i: FV) -> @location(0) vec4f {
    let sz = vec2f(textureDimensions(f_pos));
    let q = vec2i(min(floor(i.pos.xy / fu.frame.xy * sz), sz - 1.0));
    let p = textureLoad(f_pos, q, 0);
    if (p.a <= 0.0) { return vec4f(0.0); }
    let w = projection_weight(p.xyz, textureLoad(f_nrm, q, 0).xyz, fu.eye.xyz, fu.fade.xy);
    return vec4f(1.0, 0.15, 0.1, (1.0 - w) * fu.eye.w);
}
"""

BAKE_VIS = """
struct U { view_proj: mat4x4f, light_view_proj: mat4x4f, inv_view_proj: mat4x4f, eye: vec4f };
struct B { color: vec4f, mode: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var<uniform> b: B;
struct VOut { @builtin(position) pos: vec4f, @location(0) world: vec3f, @location(1) n: vec3f, @location(2) flag: f32 };
@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) flag: f32) -> VOut {
    var o: VOut;
    o.pos = u.view_proj * vec4f(p, 1.0);
    o.world = p;
    o.n = n;
    o.flag = flag;
    return o;
}
@fragment
fn fs(i: VOut) -> @location(0) vec4f {
    if (b.mode.x >= 2.5) { return b.color; }
    let v = normalize(u.eye.xyz - i.world);
    let lit = 0.25 + 0.75 * abs(dot(normalize(i.n), v));
    if (b.mode.x >= 1.5) { return vec4f(b.color.rgb * (0.6 + 0.4 * lit), b.color.a); }
    var c = b.color.rgb;
    if (b.mode.x > 0.5 && i.flag > 0.5) { c = vec3f(0.9, 0.12, 0.08); }
    return vec4f(c * lit, 1.0);
}
"""
