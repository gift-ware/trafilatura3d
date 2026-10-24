"""The 3D View's GPU Pathtracer: progressive path tracing in a compute shader.

- Geometry: the visible meshes as triangles (displaced on the CPU from the height channel), in a
  BVH over Morton-ordered triangles (an implicit balanced tree, built vectorized in numpy).
- Materials: the rasterizer's slot textures packed into PACKS textures per material (at most MAX_MATERIALS), read
  into the shaders' channels and turned into lobes by the rasterizer's `SURFACE` (one shader model for both
  renderers): the reflection lobes evaluated by `shade_light`, sampled as a mixture of cosine (diffuse,
  subsurface, fuzz), GGX (specular) and GGX (coat); transmission as a GGX refraction with Beer absorption over the
  distance travelled inside (transmission depth; a tint per crossing at depth 0), thin walled = straight through.
  Fallbacks: subsurface = wrapped diffuse at the hit (no random walk), fuzz sampled by the cosine lobe, Lambert lit
  by the point lights only (no image based light), Unlit / Mesh Info emit their value.
- Opacity by the material's blending mode, as the rasterizer: Opaque ignores it, Alpha test cuts out, Alpha blend is
  a stochastic cut-out (a ray passes with probability 1 - opacity: the average is the blend, in any order).
- Light: the environment with importance sampling (a luminance CDF of the panorama) and multiple
  importance sampling with the BSDF, the two point lights, Russian roulette.
- Ground plane: a shadow catcher over the background with shadow intensity, local lights and glossy
  reflections of the meshes.
- Accumulation in a float buffer, shown through the same exposure / tone mapping as the rasterizer.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import wgpu

from trafilatura3d.bvh import LEAF, bvh_bounds, morton_bvh  # noqa: F401 - re-exported for tests
from trafilatura3d.editor.render3d import COMMON, ENV_H, ENV_W, SURFACE, UNLIT, material_uniform

MAX_MATERIALS = 3
CDF_BLOCK = 4  # importance sampling grid: the sharp level 0 averaged in 4 x 4 blocks (256 x 128)
PACK_MAX = 2048
# The packs of a material, in the order of SURFACE's `Ch` (11 vec4): per component (slot, its component).
_rgb = lambda slot: ((slot, 0), (slot, 1), (slot, 2))
PACKS = (
    (*_rgb("baseColor"), ("opacity", 0)), (*_rgb("normal"), ("height", 0)),
    (("roughness", 0), ("metallic", 0), ("ambientOcclusion", 0), ("specularWeight", 0)),
    (*_rgb("emissive"), ("anisotropyLevel", 0)), (*_rgb("specularColor"), ("anisotropyAngle", 0)),
    (*_rgb("coatColor"), ("coatWeight", 0)), (*_rgb("coatNormal"), ("coatRoughness", 0)),
    (*_rgb("fuzzColor"), ("fuzzWeight", 0)), (*_rgb("subsurfaceColor"), ("subsurfaceWeight", 0)),
    (*_rgb("transmissionColor"), ("transmissionWeight", 0)),
    (("fuzzRoughness", 0), ("subsurfaceRadius", 0), ("subsurfaceRadius", 0), ("subsurfaceRadius", 0)),
)
M_SIZE = 240  # bytes of SURFACE's `M` (its fields, then the library shaders' `custom`)

# One pack: four sources (the same texture for the components of a color), a component of each.
PACK = """
@group(0) @binding(0) var<uniform> sel: vec4u;
@group(0) @binding(1) var t0: texture_2d<f32>;
@group(0) @binding(2) var t1: texture_2d<f32>;
@group(0) @binding(3) var t2: texture_2d<f32>;
@group(0) @binding(4) var t3: texture_2d<f32>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var out: texture_storage_2d<rgba16float, write>;
fn at(t: texture_2d<f32>, uv: vec2f, k: u32) -> f32 { return textureSampleLevel(t, samp, uv, 0.0)[k]; }
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(out));
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    textureStore(out, c, vec4f(at(t0, uv, sel.x), at(t1, uv, sel.y), at(t2, uv, sel.z), at(t3, uv, sel.w)));
}
"""

MATERIAL_BINDINGS = "".join(f"@group(1) @binding({mi * len(PACKS) + k}) var m{mi}_{k}: texture_2d<f32>;\n"
                            for mi in range(MAX_MATERIALS) for k in range(len(PACKS)))
MTEX_CASES = "".join(f"        case {mi * len(PACKS) + k}u: {{ return textureSampleLevel(m{mi}_{k}, matsamp, uv, 0.0); }}\n"
                     for mi in range(MAX_MATERIALS) for k in range(len(PACKS)))

TRACE = COMMON + SURFACE + MATERIAL_BINDINGS + f"const PACKS = {len(PACKS)}u;\nconst UNLIT = {UNLIT}u;\n" + """
struct P {
    inv_view_proj: mat4x4f,
    eye: vec4f,
    frame: u32, leaf_base: u32, tri_count: u32, max_bounces: u32,
    width: u32, height: u32, env_w: u32, env_h: u32,
    env: vec4f,          // rotation, intensity, background mip level, environment total weight
    ground: vec4f,       // enabled, height, shadow intensity, local lights
    reflect: vec4f,      // enabled, opacity, roughness, opengl normals
    p0_pos: vec4f, p0_col: vec4f, p1_pos: vec4f, p1_col: vec4f,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;  // per vertex: position + u, normal + v, tangent, color
@group(0) @binding(2) var<storage, read> tris: array<vec4u>;
@group(0) @binding(3) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(4) var<storage, read> cdf: array<f32>;
@group(0) @binding(5) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(6) var env: texture_2d<f32>;
@group(0) @binding(7) var lin_samp: sampler;
@group(0) @binding(8) var matsamp: sampler;
@group(0) @binding(9) var<storage, read> mats: array<M>;
@group(0) @binding(10) var brdf: texture_2d<f32>;

var<private> seed: u32;
fn rnd() -> f32 {
    seed = seed * 747796405u + 2891336453u;
    var w = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    w = (w >> 22u) ^ w;
    return f32(w) / 4294967296.0;
}

fn mtex(mi: u32, k: u32, uv: vec2f) -> vec4f {
    switch mi * PACKS + k {
""" + MTEX_CASES + """        default: { return vec4f(0.0); }
    }
}

fn rot_y(d: vec3f, a: f32) -> vec3f {
    return vec3f(d.x * cos(a) - d.z * sin(a), d.y, d.x * sin(a) + d.z * cos(a));
}
fn env_radiance(d: vec3f, level: f32) -> vec3f {
    return textureSampleLevel(env, lin_samp, dir_to_uv(rot_y(d, p.env.x)), level).rgb * p.env.y;
}

// --- environment importance sampling: marginal (rows) then conditional (columns) CDFs -------------
fn find(start: u32, count: u32, x: f32) -> u32 {
    var lo = 0u;
    var hi = count - 1u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (cdf[start + mid] < x) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}
fn env_pdf_uv(uv: vec2f) -> f32 {
    let w = p.env_w;
    let h = p.env_h;
    let x = min(u32(uv.x * f32(w)), w - 1u);
    let y = min(u32(uv.y * f32(h)), h - 1u);
    let row = h + y * w;  // conditional CDF of row y starts after the marginal
    let pr = cdf[y] - select(0.0, cdf[y - 1u], y > 0u);
    let pc = cdf[row + x] - select(0.0, cdf[row + x - 1u], x > 0u);
    return pr * pc * f32(w * h);
}
fn env_pdf(d: vec3f) -> f32 {  // solid angle pdf of a world direction
    let e = rot_y(d, p.env.x);
    let uv = dir_to_uv(e);
    let st = max(sqrt(1.0 - e.y * e.y), 1e-4);
    return env_pdf_uv(uv) / (2.0 * PI * PI * st);
}
fn env_sample() -> vec3f {  // a world direction
    let y = find(0u, p.env_h, rnd());
    let x = find(p.env_h + y * p.env_w, p.env_w, rnd());
    let uv = (vec2f(f32(x), f32(y)) + vec2f(rnd(), rnd())) / vec2f(f32(p.env_w), f32(p.env_h));
    return rot_y(uv_to_dir(uv), -p.env.x);
}

// --- geometry ------------------------------------------------------------------------------------
struct Hit { t: f32, tri: u32, b: vec2f };

fn vpos(i: u32) -> vec3f { return verts[i * 4u].xyz; }

fn box_hit(o: vec3f, inv: vec3f, lo: vec3f, hi: vec3f, tmax: f32) -> f32 {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return select(1e30, max(tn, 0.0), tn <= tf && tf > 0.0 && tn < tmax);
}

fn mesh_uv(tri: u32, b: vec2f) -> vec2f {
    let t = tris[tri];
    let w = 1.0 - b.x - b.y;
    let u0 = vec2f(verts[t.x * 4u].w, verts[t.x * 4u + 1u].w);
    let u1 = vec2f(verts[t.y * 4u].w, verts[t.y * 4u + 1u].w);
    let u2 = vec2f(verts[t.z * 4u].w, verts[t.z * 4u + 1u].w);
    return u0 * w + u1 * b.x + u2 * b.y;
}
fn uv_of(tri: u32, b: vec2f) -> vec2f {
    let m = mats[tris[tri].w];
    return mesh_uv(tri, b) * m.tiling * vec2f(m.uv_u, m.uv_v);
}

// Closest hit (any_hit: stop at the first), through the opacity by the blending mode: Opaque ignores it, Alpha test
// cuts out below the threshold, Alpha blend lets a ray through with probability 1 - opacity (partial transmission,
// straight on, untinted: shadows and reflections included).
fn trace(o: vec3f, d: vec3f, tmax: f32, any_hit: bool) -> Hit {
    var best = Hit(tmax, 0xffffffffu, vec2f(0.0));
    let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
    var stack: array<u32, 40>;
    var sp = 0u;
    stack[0] = 0u;
    sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp -= 1u;
        let n = stack[sp];
        if (box_hit(o, inv, nodes[n * 2u].xyz, nodes[n * 2u + 1u].xyz, best.t) >= best.t) { continue; }
        if (n >= p.leaf_base) {
            let first = (n - p.leaf_base) * 4u;
            for (var k = 0u; k < 4u; k++) {
                let ti = first + k;
                if (ti >= p.tri_count) { break; }
                let t = tris[ti];
                let a = vpos(t.x);
                let e1 = vpos(t.y) - a;
                let e2 = vpos(t.z) - a;
                let q = cross(d, e2);
                let det = dot(e1, q);
                if (abs(det) < 1e-10) { continue; }
                let f = 1.0 / det;
                let s = o - a;
                let bu = dot(s, q) * f;
                if (bu < 0.0 || bu > 1.0) { continue; }
                let r = cross(s, e1);
                let bv = dot(d, r) * f;
                if (bv < 0.0 || bu + bv > 1.0) { continue; }
                let tt = dot(e2, r) * f;
                if (tt <= 1e-4 || tt >= best.t) { continue; }
                let m = mats[t.w];
                if (m.blending != 0u) {
                    let alpha = mtex(t.w, 0u, uv_of(ti, vec2f(bu, bv))).a;
                    if (m.blending == ALPHA_TEST) {
                        if (alpha < m.alpha_threshold) { continue; }
                    } else if (rnd() >= alpha) { continue; }
                }
                best = Hit(tt, ti, vec2f(bu, bv));
                if (any_hit) { return best; }
            }
        } else if (sp + 2u <= 40u) {
            stack[sp] = n * 2u + 2u;
            stack[sp + 1u] = n * 2u + 1u;
            sp += 2u;
        }
    }
    return best;
}

struct Surface { pos: vec3f, fr: Frame, L: Lobes, unlit: bool, emissive: vec3f, front: bool };

fn tangent_normal(rgb: vec3f, scale: f32, t: vec3f, bv: vec3f, n: vec3f) -> vec3f {
    var tn = rgb * 2.0 - 1.0;
    if (p.reflect.w == 0.0) { tn.y = -tn.y; }
    if (scale > 1.0) {
        tn = normalize(vec3f(tn.xy, tn.z / scale));
    } else if (scale < 1.0) {
        tn = normalize(mix(vec3f(0.0, 0.0, 1.0), normalize(tn), max(scale, 0.0)));
    }
    return normalize(t * tn.x - bv * tn.y + n * tn.z);
}
fn facing(n: vec3f, ng: vec3f, d: vec3f) -> vec3f {  // a shading normal kept on the viewer's side
    if (dot(n, -d) < 0.0) { return normalize(n + ng * (0.01 - dot(n, -d))); }
    return n;
}

fn surface(h: Hit, o: vec3f, d: vec3f) -> Surface {
    let t = tris[h.tri];
    let w = 1.0 - h.b.x - h.b.y;
    let i0 = t.x * 4u; let i1 = t.y * 4u; let i2 = t.z * 4u;
    var s: Surface;
    s.pos = o + d * h.t;
    var ng = normalize(cross(verts[i1].xyz - verts[i0].xyz, verts[i2].xyz - verts[i0].xyz));
    var n = normalize(verts[i0 + 1u].xyz * w + verts[i1 + 1u].xyz * h.b.x + verts[i2 + 1u].xyz * h.b.y);
    let tg = verts[i0 + 2u] * w + verts[i1 + 2u] * h.b.x + verts[i2 + 2u] * h.b.y;
    if (dot(n, ng) < 0.0) { ng = -ng; }
    s.front = dot(ng, d) < 0.0;  // the outside of the mesh (its vertex normals')
    if (!s.front) { ng = -ng; n = -n; }
    let uv = uv_of(h.tri, h.b);
    let mi = t.w;
    let m = mats[mi];
    var c: Ch;
    let a0 = mtex(mi, 0u, uv); c.base = a0.rgb; c.opacity = a0.a;
    let a1 = mtex(mi, 1u, uv); c.nrm = a1.rgb; c.height = a1.a;
    let a2 = mtex(mi, 2u, uv); c.rough = a2.r; c.metal = a2.g; c.ao = a2.b; c.spec_weight = a2.a;
    let a3 = mtex(mi, 3u, uv); c.emit = a3.rgb; c.aniso = a3.a;
    let a4 = mtex(mi, 4u, uv); c.spec_color = a4.rgb; c.aniso_angle = a4.a;
    let a5 = mtex(mi, 5u, uv); c.coat_color = a5.rgb; c.coat = a5.a;
    let a6 = mtex(mi, 6u, uv); c.coat_nrm = a6.rgb; c.coat_rough = a6.a;
    let a7 = mtex(mi, 7u, uv); c.fuzz_color = a7.rgb; c.fuzz = a7.a;
    let a8 = mtex(mi, 8u, uv); c.sss_color = a8.rgb; c.sss = a8.a;
    let a9 = mtex(mi, 9u, uv); c.trans_color = a9.rgb; c.trans = a9.a;
    let a10 = mtex(mi, 10u, uv); c.fuzz_rough = a10.r; c.sss_radius = a10.g;
    s.L = lobes(m, c);
    let tt = normalize(tg.xyz - n * dot(n, tg.xyz));
    let bv = cross(n, tt) * select(-1.0, 1.0, tg.w >= 0.0);
    s.fr.ng = ng;
    s.fr.n = facing(tangent_normal(c.nrm, m.normal_scale, tt, bv, n), ng, d);
    s.fr.cn = facing(tangent_normal(c.coat_nrm, m.coat_normal_scale, tt, bv, n), ng, d);
    let ta = tt * cos(s.L.angle) - bv * sin(s.L.angle);
    s.fr.t = normalize(ta - s.fr.n * dot(s.fr.n, ta));
    s.fr.b = cross(s.fr.n, s.fr.t);
    s.unlit = s.L.model == MODEL_UNLIT || s.L.model == MODEL_MESH_INFO;
    s.emissive = emitted(s.L, max(dot(s.fr.n, -d), 1e-4));
    if (s.L.model == MODEL_UNLIT) { s.emissive = s.L.albedo; }  // lin(c.base), or an unlit library shader's color
    if (s.L.model == MODEL_MESH_INFO) {
        let col = verts[i0 + 3u] * w + verts[i1 + 3u] * h.b.x + verts[i2 + 3u] * h.b.y;
        s.emissive = lin(mesh_info(m.mode, n, tt, bv, mesh_uv(h.tri, h.b), col, s.pos));
    }
    return s;
}

// --- BSDF: the reflection of SURFACE sampled as a mixture of cosine and two GGX lobes ------------------------------
fn ggx_d(nh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let k = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * k * k);
}
fn lobe_weights(s: Surface, nv: f32) -> vec3f {  // diffuse, specular, coat sampling probabilities
    let L = s.L;
    if (L.model == MODEL_BLINN) {
        let ws = lum3(L.tint);
        let wd = lum3(L.albedo);
        return vec3f(wd, ws, 0.0) / max(wd + ws, 1e-4);
    }
    let fs = mix(spec_fresnel(L, nv), metal_fresnel(L, nv), L.metal);
    let ws = max(lum3(fs), 0.02);
    let wd = (1.0 - L.metal) * (1.0 - L.trans) * (lum3(L.albedo) + L.sss + L.fuzz) * (1.0 - ws);
    let wc = L.coat * coat_fresnel(L, nv);
    let sum = max(ws + wd + wc, 1e-4);
    var w = vec3f(wd, ws, wc) / sum;
    w = select(w, max(w, vec3f(0.05)), w > vec3f(0.0));
    return w / (w.x + w.y + w.z);
}
fn bsdf_eval(s: Surface, v: vec3f, l: vec3f) -> vec4f {  // rgb = f * cos, a = pdf
    let n = s.fr.n;
    let nl = dot(n, l);
    if (dot(s.fr.ng, l) <= 0.0) { return vec4f(0.0); }
    let nv = max(dot(n, v), 1e-4);
    let w = lobe_weights(s, nv);
    let h = normalize(l + v);
    let vh = max(dot(v, h), 1e-4);
    let rb = max(coated_roughness(s.L), 0.03);
    var pdf = w.x * max(nl, 0.0) / PI + w.y * ggx_d(max(dot(n, h), 0.0), rb * rb) * max(dot(n, h), 0.0) / (4.0 * vh);
    if (w.z > 0.0) {
        let cn = s.fr.cn;
        let rc = max(s.L.coat_rough, 0.03);
        pdf += w.z * ggx_d(max(dot(cn, h), 0.0), rc * rc) * max(dot(cn, h), 0.0) / (4.0 * vh);
    }
    if (pdf <= 0.0) { return vec4f(0.0); }
    return vec4f(shade_light(s.L, s.fr, v, l, vec3f(1.0)), pdf);
}
fn bsdf_sample(s: Surface, v: vec3f) -> vec3f {
    let w = lobe_weights(s, max(dot(s.fr.n, v), 1e-4));
    let x = rnd();
    if (x < w.y) {
        let rb = max(coated_roughness(s.L), 0.03);
        return reflect(-v, ggx_half(vec2f(rnd(), rnd()), s.fr.n, rb * rb));
    }
    if (x < w.y + w.z) {
        let rc = max(s.L.coat_rough, 0.03);
        return reflect(-v, ggx_half(vec2f(rnd(), rnd()), s.fr.cn, rc * rc));
    }
    let r = sqrt(rnd());
    let phi = TAU * rnd();
    return normalize(basis(s.fr.n) * vec3f(r * cos(phi), r * sin(phi), sqrt(max(1.0 - r * r, 0.0))));
}
// Energy and tint of the transmission lobe (OpenPBR: the dielectric's share past the specular, under coat and fuzz).
fn transmission(s: Surface, nv: f32) -> vec4f {
    let L = s.L;
    if (L.model != MODEL_LAYERED || L.trans <= 0.0) { return vec4f(0.0); }
    let rb = coated_roughness(L);
    let e = lum3(spec_albedo(spec_fresnel(L, 1.0), vec3f(1.0), nv, rb));
    var k = (1.0 - L.metal) * L.trans * (1.0 - e);
    k *= mix(1.0, 1.0 - coat_fresnel(L, nv), L.coat) * (1.0 - L.fuzz * fuzz_albedo(nv, L.fuzz_rough));
    var tint = L.trans_color;
    if (L.thin == 0.0 && L.depth > 0.0) { tint = vec3f(1.0); }  // absorbed along the way inside instead
    else if (L.thin == 0.0) { tint = sqrt(L.trans_color); }      // a tint at each crossing
    return vec4f(tint, k);
}
fn power_heuristic(a: f32, b: f32) -> f32 { return a * a / max(a * a + b * b, 1e-12); }

fn point_light(pos: vec4f, col: vec4f, s: Surface, v: vec3f) -> vec3f {
    if (pos.w == 0.0) { return vec3f(0.0); }
    let dl = pos.xyz - s.pos;
    let dist = length(dl);
    let l = dl / dist;
    if (dot(s.fr.ng, l) <= 0.0 && s.L.sss <= 0.0) { return vec3f(0.0); }
    let f = shade_light(s.L, s.fr, v, l, vec3f(1.0));
    if (all(f <= vec3f(0.0))) { return vec3f(0.0); }
    if (trace(s.pos + s.fr.ng * 1e-3, l, dist, true).tri != 0xffffffffu) { return vec3f(0.0); }
    return f * col.rgb * col.w / (dist * dist);
}

// Direct light at a surface: one environment sample (MIS) and the point lights (Lambert: the point lights only).
fn direct(s: Surface, v: vec3f) -> vec3f {
    var c = vec3f(0.0);
    if (s.L.model != MODEL_LAMBERT) {
        let l = env_sample();
        let f = bsdf_eval(s, v, l);
        if (f.a > 0.0) {
            let pl = env_pdf(l);
            if (pl > 0.0 && trace(s.pos + s.fr.ng * 1e-3, l, 1e30, true).tri == 0xffffffffu) {
                c += f.rgb * env_radiance(l, 0.0) / pl * power_heuristic(pl, f.a);
            }
        }
    }
    c += point_light(p.p0_pos, p.p0_col, s, v);
    c += point_light(p.p1_pos, p.p1_col, s, v);
    return c;
}

// A full path from a surface; env hits after a BSDF bounce are weighted by MIS (not after a refraction).
fn path(first: Surface, v0: vec3f) -> vec3f {
    var s = first;
    var v = v0;
    var color = vec3f(0.0);
    var through = vec3f(1.0);
    var absorb = vec3f(0.0);  // the medium the ray travels in: transmission color, depth (0 = none)
    var depth = 0.0;
    for (var bounce = 0u; bounce < p.max_bounces; bounce++) {
        if (s.unlit) { color += through * s.emissive; break; }
        let nv = max(dot(s.fr.n, v), 1e-4);
        let tr = transmission(s, nv);
        var l: vec3f;
        var refracted = false;
        if (!s.front && tr.w > 0.0) {  // leaving a transmissive volume
            refracted = true;
        } else {
            var c = through * (s.emissive + direct(s, v));
            if (bounce > 0u) { c = min(c, vec3f(8.0)); }  // indirect fireflies
            color += c;
            if (s.L.model == MODEL_LAMBERT) { break; }  // no image based light, no bounces
            refracted = tr.w > 0.0 && rnd() < clamp(tr.w, 0.0, 0.95);
            if (refracted) { through *= tr.rgb * tr.w / clamp(tr.w, 0.0, 0.95); }
        }
        if (refracted) {
            let rb = max(coated_roughness(s.L), 0.03);
            if (s.L.thin > 0.0) {
                l = -v;
            } else {
                let hm = ggx_half(vec2f(rnd(), rnd()), s.fr.n, rb * rb);
                let eta = select(s.L.eta, 1.0 / s.L.eta, s.front);
                l = refract(-v, hm, eta);
                if (dot(l, l) < 1e-6) { l = reflect(-v, hm); }  // total internal reflection
                if (!s.front && s.L.depth <= 0.0) { through *= sqrt(s.L.trans_color); }
                if (s.front && s.L.depth > 0.0) { absorb = s.L.trans_color; depth = s.L.depth; }
                if (!s.front) { depth = 0.0; }
            }
        } else {
            l = bsdf_sample(s, v);
            let f = bsdf_eval(s, v, l);
            if (f.a <= 1e-6) { break; }
            let keep = 1.0 - select(0.0, clamp(tr.w, 0.0, 0.95), tr.w > 0.0);
            through *= f.rgb / f.a / max(keep, 0.05);
        }
        if (bounce >= 2u) {
            let q = clamp(max(through.r, max(through.g, through.b)), 0.05, 0.95);
            if (rnd() > q) { break; }
            through /= q;
        }
        let side = select(s.fr.ng, -s.fr.ng, dot(l, s.fr.ng) < 0.0);
        let h = trace(s.pos + side * 1e-3, l, 1e30, false);
        if (depth > 0.0 && h.tri != 0xffffffffu) {  // Beer: transmission color at the transmission depth
            through *= pow(max(absorb, vec3f(1e-4)), vec3f(h.t / depth));
        }
        if (h.tri == 0xffffffffu) {
            var wmis = 1.0;
            if (!refracted) { wmis = power_heuristic(bsdf_eval(s, v, l).a, env_pdf(l)); }
            color += min(through * env_radiance(l, 0.0) * wmis, vec3f(8.0));
            break;
        }
        s = surface(h, s.pos + side * 1e-3, l);
        v = -l;
    }
    return color;
}

// Ground plane: the background darkened where the meshes block the environment (and the point
// lights, with local lights), plus glossy reflections of the meshes.
fn ground(o: vec3f, d: vec3f, t: f32, bg: vec3f) -> vec3f {
    let pos = o + d * t;
    let up = vec3f(0.0, 1.0, 0.0);
    var occluded = 0.0;
    accum_weight = 0.0;
    let l = env_sample();
    if (l.y > 0.0) {
        let w = lum3(env_radiance(l, 0.0)) * l.y / max(env_pdf(l), 1e-6);
        let blocked = trace(pos + up * 1e-3, l, 1e30, true).tri != 0xffffffffu;
        occluded = select(0.0, 1.0, blocked) * w;
        accum_weight = w;
    }
    var c = bg;
    if (p.ground.w > 0.0) {
        for (var k = 0u; k < 2u; k++) {
            let lp = select(p.p0_pos, p.p1_pos, k == 1u);
            if (lp.w == 0.0) { continue; }
            let dl = lp.xyz - pos;
            let dist = length(dl);
            if (trace(pos + up * 1e-3, dl / dist, dist, true).tri != 0xffffffffu) {
                c *= 1.0 - 0.5 * p.ground.z;
            }
        }
    }
    let ratio = select(0.0, occluded / max(accum_weight, 1e-6), accum_weight > 0.0);
    c *= 1.0 - p.ground.z * ratio;
    if (p.reflect.x > 0.0) {
        let h = ggx_half(vec2f(rnd(), rnd()), up, max(p.reflect.z, 0.02) * max(p.reflect.z, 0.02));
        let r = reflect(d, h);
        if (r.y > 0.0) {
            let hit = trace(pos + up * 1e-3, r, 1e30, false);
            if (hit.tri != 0xffffffffu) {
                let s = surface(hit, pos + up * 1e-3, r);
                let lit = select(s.emissive + direct(s, -r), s.emissive, s.unlit);
                c = mix(c, lit, p.reflect.y);
            }
        }
    }
    return c;
}
var<private> accum_weight: f32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= p.width || id.y >= p.height) { return; }
    let pix = id.y * p.width + id.x;
    seed = (pix * 9781u + p.frame * 6271u) ^ 0x9e3779b9u;
    _ = rnd();
    let jitter = vec2f(rnd(), rnd());
    let ndc = vec2f((f32(id.x) + jitter.x) / f32(p.width) * 2.0 - 1.0, 1.0 - (f32(id.y) + jitter.y) / f32(p.height) * 2.0);
    let far = p.inv_view_proj * vec4f(ndc, 1.0, 1.0);
    let near = p.inv_view_proj * vec4f(ndc, 0.0, 1.0);
    let o = near.xyz / near.w;
    let d = normalize(far.xyz / far.w - o);
    let bg = env_radiance(d, p.env.z);
    var color: vec3f;
    let h = trace(o, d, 1e30, false);
    var tg = 1e30;
    if (p.ground.x > 0.0 && d.y < 0.0) { tg = (p.ground.y - o.y) / d.y; }
    if (h.tri != 0xffffffffu && h.t < tg) {
        color = path(surface(h, o, d), -d);
    } else if (tg < 1e29 && length((o + d * tg).xz) < 8.0) {
        let fade = 1.0 - smoothstep(1.8, 5.5, length((o + d * tg).xz));
        color = mix(bg, ground(o, d, tg, bg), fade);
    } else {
        color = bg;
    }
    if (any(color != color)) { color = vec3f(0.0); }  // NaN guard
    color = min(color, vec3f(64.0));  // fireflies
    accum[pix] += vec4f(color, 1.0);
}
"""

DISPLAY = """
struct D { exposure: f32, tonemap: u32, channel: u32, linear: u32, width: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> dp: D;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var ocio_lut: texture_3d<f32>;
@group(0) @binding(3) var clamp_samp: sampler;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    return vec4f(p, 0.0, 1.0);
}
fn hable(x: vec3f) -> vec3f {
    let a = 0.15; let b = 0.50; let c = 0.10; let d = 0.20; let e = 0.02; let f = 0.30;
    return (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f;
}
fn linear_to_srgb(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let a = accum[u32(frag.y) * dp.width + u32(frag.x)];
    let x = max(a.rgb / max(a.w, 1.0), vec3f(0.0)) * exp2(dp.exposure);
    if (dp.linear == 1u) { return vec4f(x, 1.0); }
    if (dp.tonemap == 3u) {  // OCIO view (log2 shaped LUT, same range as the rasterizer)
        let t = clamp((log2(max(x, vec3f(1e-6))) + 12.0) / 20.0, vec3f(0.0), vec3f(1.0));
        let n = f32(textureDimensions(ocio_lut).x);
        var o = textureSampleLevel(ocio_lut, clamp_samp, t * ((n - 1.0) / n) + 0.5 / n, 0.0).rgb;
        switch dp.channel {
            case 1u: { o = vec3f(o.r); }
            case 2u: { o = vec3f(o.g); }
            case 3u: { o = vec3f(o.b); }
            default: {}
        }
        return vec4f(o, 1.0);
    }
    var m: vec3f;
    switch dp.tonemap {
        case 1u: { m = hable(x * 2.0) / hable(vec3f(11.2)); }
        case 2u: { m = x; }
        default: { m = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14); }
    }
    var c = linear_to_srgb(clamp(m, vec3f(0.0), vec3f(1.0)));
    switch dp.channel {
        case 1u: { c = vec3f(c.r); }
        case 2u: { c = vec3f(c.g); }
        case 3u: { c = vec3f(c.b); }
        default: {}
    }
    return vec4f(c, 1.0);
}
"""


def displace(objects, renderer, height_scale, height_level):
    """Copies of the visible objects with positions moved along the normals by the height channel
    (bilinear, wrapping), as the rasterizer's vertex displacement."""
    out = []
    cache = {}
    for o in objects:
        if not o.visible:
            continue
        pos = o.pos.copy()
        if height_scale:
            for s in o.submeshes:
                tri = o.idx[s.start:s.start + s.count]
                verts = np.unique(tri)
                if s.material not in cache:
                    cache[s.material] = _read_height(renderer, s.material)
                h, tiling = cache[s.material]
                uv = o.uv[verts] * tiling
                pos[verts] += o.nrm[verts] * ((_bilinear(h, uv) - height_level) * height_scale)[:, None]
        out.append((o, pos))
    return out


def _read_height(renderer, material):
    """The height slot and the UV scale the shaders apply (Tiling x UV scale)."""
    img = renderer.read_material(material, "height")[..., 0]
    uniforms = renderer.materials[min(material, len(renderer.materials) - 1)]["spec"].get("uniforms", {})
    return img, uniforms.get("tiling", 1.0) * np.asarray(uniforms.get("uv_scale", (1.0, 1.0)), np.float32)


def _bilinear(img, uv):
    h, w = img.shape
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    g = lambda yy, xx: img[yy % h, xx % w]
    return ((g(y0, x0) * (1 - fx) + g(y0, x0 + 1) * fx) * (1 - fy) + (g(y0 + 1, x0) * (1 - fx) + g(y0 + 1, x0 + 1) * fx) * fy)


class PathTracer:
    def __init__(self, renderer):
        self.r = renderer
        self.device = d = renderer.device
        self.trace_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": d.create_shader_module(code=TRACE), "entry_point": "main"})
        self.pack_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": d.create_shader_module(code=PACK), "entry_point": "main"})
        self.uniform = d.create_buffer(size=512, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.display_uniform = d.create_buffer(size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._display = {}
        self.accum = None
        self.size = (0, 0)
        self.samples = 0
        self.key = None
        self.scene_version = 0
        self.geometry = None
        self.material_group = None
        self.cdf = None
        self.env_version = None

    # --- scene ------------------------------------------------------------------------------------

    def set_scene(self, objects, look):
        """Triangles of the visible meshes (displaced), their BVH and the packed materials."""
        height = look.displacement if look.enable_displacement else 0.0
        parts = displace(objects, self.r, height, look.height_level)
        if not parts:
            self.geometry = None
            return
        vbase, verts, tris = 0, [], []
        for o, pos in parts:
            v = np.zeros((len(pos), 4, 4), np.float32)
            v[:, 0, :3], v[:, 0, 3] = pos, o.uv[:, 0]
            v[:, 1, :3], v[:, 1, 3] = o.nrm, o.uv[:, 1]
            v[:, 2] = o.tan
            colors = getattr(o, "colors", None)  # Mesh Info's vertex color
            v[:, 3] = 1.0 if colors is None else colors
            verts.append(v.reshape(-1, 4))
            for s in o.submeshes:
                t = o.idx[s.start:s.start + s.count].reshape(-1, 3).astype(np.uint32) + vbase
                mat = np.full((len(t), 1), min(s.material, MAX_MATERIALS - 1), np.uint32)
                tris.append(np.concatenate([t, mat], 1))
            vbase += len(pos)
        verts = np.concatenate(verts)
        tris = np.concatenate(tris)
        positions = verts.reshape(-1, 4, 4)[:, 0, :3]
        tri_pos = positions[tris[:, :3]]
        order, leaves = morton_bvh(tri_pos.mean(1))
        tris, tri_pos = tris[order], tri_pos[order]
        nodes = bvh_bounds(tri_pos, leaves)
        mk = lambda a: self.device.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(),
                                                           usage=wgpu.BufferUsage.STORAGE)
        self.geometry = (mk(verts), mk(tris), mk(nodes), len(tris), leaves - 1)
        self._pack_materials()
        self.scene_version += 1

    def _pack_materials(self):
        """Each material's slots into the PACKS textures (each as large as its largest source, up to PACK_MAX: a
        pack of constants stays 1x1) and the materials' uniforms into a storage buffer."""
        d = self.device
        usage = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
        views = []
        enc = d.create_command_encoder()
        mats = self.r.materials
        uniforms = b""
        for i in range(MAX_MATERIALS):
            m = mats[min(i, len(mats) - 1)]
            uniforms += material_uniform(m["spec"])
            for pack in PACKS:
                sources = [m["textures"][slot] for slot, _ in pack]
                size = min(max(max(t.size[0], t.size[1]) for t in sources), PACK_MAX)
                out = d.create_texture(size=(size, size, 1), format="rgba16float", usage=usage)
                sel = d.create_buffer_with_data(data=struct.pack("4I", *(k for _, k in pack)),
                                                usage=wgpu.BufferUsage.UNIFORM)
                group = d.create_bind_group(layout=self.pack_pipe.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": {"buffer": sel}}] + [
                    {"binding": 1 + k, "resource": t.create_view()} for k, t in enumerate(sources)] + [
                    {"binding": 5, "resource": self.r.mat_sampler}, {"binding": 6, "resource": out.create_view()}])
                cp = enc.begin_compute_pass()
                cp.set_pipeline(self.pack_pipe)
                cp.set_bind_group(0, group)
                cp.dispatch_workgroups((size + 7) // 8, (size + 7) // 8)
                cp.end()
                views.append(out.create_view())
        d.queue.submit([enc.finish()])
        self.material_group = d.create_bind_group(layout=self.trace_pipe.get_bind_group_layout(1), entries=[
            {"binding": k, "resource": v} for k, v in enumerate(views)])
        self.material_buffer = d.create_buffer_with_data(data=uniforms, usage=wgpu.BufferUsage.STORAGE)

    def _environment_cdf(self):
        """Marginal + conditional CDFs of the environment's luminance x sin(theta)."""
        t = self.r.env  # level 0: the other levels are GGX prefiltered, too blurred for a sun disc
        data = self.device.queue.read_texture({"texture": t}, {"bytes_per_row": ENV_W * 8}, (ENV_W, ENV_H, 1))
        full = np.frombuffer(data, np.float16).reshape(ENV_H, ENV_W, 4)[..., :3].astype(np.float64)
        w, h = ENV_W // CDF_BLOCK, ENV_H // CDF_BLOCK
        img = full.reshape(h, CDF_BLOCK, w, CDF_BLOCK, 3).max((1, 3))  # max keeps small bright sources
        lum = img @ np.array([0.2126, 0.7152, 0.0722])
        theta = (np.arange(h) + 0.5) / h * math.pi
        weight = lum * np.sin(theta)[:, None] + 1e-8
        rows = weight.sum(1)
        marginal = np.cumsum(rows) / rows.sum()
        conditional = np.cumsum(weight, 1) / rows[:, None]
        cdf = np.concatenate([marginal, conditional.reshape(-1)]).astype(np.float32)
        self.cdf = (self.device.create_buffer_with_data(data=cdf.tobytes(), usage=wgpu.BufferUsage.STORAGE), w, h)

    # --- frames -----------------------------------------------------------------------------------

    def reset(self):
        self.samples = 0
        if self.accum is not None:
            self.device.queue.write_buffer(self.accum, 0, bytes(self.size[0] * self.size[1] * 16))

    def draw(self, target, eye, view_proj, look, scene_lights, env_version, max_samples, linear=False,
             spp=1, max_bounces=6, reflections=(True, 0.5, 0.2), local_lights=True):
        """Add `spp` samples per pixel (until `max_samples`) and show the average in `target`."""
        d = self.device
        w, h = target.size[0], target.size[1]
        if (w, h) != self.size:
            self.size = (w, h)
            self.accum = d.create_buffer(size=w * h * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            self.samples = 0
        if env_version != self.env_version or self.cdf is None:
            self._environment_cdf()
            self.env_version = env_version
        lo, _ = self.r.scene_bounds()
        height = look.displacement if look.enable_displacement else 0.0
        ground_y = float(lo[1]) - abs(height) * look.height_level - 0.003 + look.ground_height
        lights = []
        for p in list(scene_lights)[:2] + [None] * (2 - len(scene_lights[:2])):
            lights += [*p[0], 1.0, *p[1], p[2]] if p else [0.0] * 8
        key = (tuple(np.round(view_proj.ravel(), 6)), look.env_rotation, look.env_intensity, look.background_blur,
               look.ground, ground_y, look.ground_shadow, look.opengl_normals, tuple(lights), self.scene_version,
               env_version, reflections, local_lights, max_bounces, w, h)
        if key != self.key:
            self.key = key
            self.reset()
        if self.geometry is not None and self.samples < max_samples:
            verts, tris, nodes, ntri, leaf_base = self.geometry
            cdf, cw, ch = self.cdf
            group = d.create_bind_group(layout=self.trace_pipe.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": self.uniform}},
                {"binding": 1, "resource": {"buffer": verts}}, {"binding": 2, "resource": {"buffer": tris}},
                {"binding": 3, "resource": {"buffer": nodes}}, {"binding": 4, "resource": {"buffer": cdf}},
                {"binding": 5, "resource": {"buffer": self.accum}},
                {"binding": 6, "resource": self.r.env.create_view()},
                {"binding": 7, "resource": self.r.lin}, {"binding": 8, "resource": self.r.mat_sampler},
                {"binding": 9, "resource": {"buffer": self.material_buffer}},
                {"binding": 10, "resource": self.r.brdf.create_view()}])
            for _ in range(min(spp, max_samples - self.samples)):
                data = np.linalg.inv(view_proj).T.astype(np.float32).tobytes() + struct.pack(
                    "4f 8I 4f 4f 4f 16f",
                    *eye, 1.0,
                    self.samples, leaf_base, ntri, max_bounces, w, h, cw, ch,
                    look.env_rotation, look.env_intensity, look.background_blur, 0.0,
                    float(look.ground), ground_y, look.ground_shadow, float(local_lights),
                    float(reflections[0]), reflections[1], reflections[2], float(look.opengl_normals),
                    *lights)
                d.queue.write_buffer(self.uniform, 0, data)
                enc = d.create_command_encoder()
                cp = enc.begin_compute_pass()
                cp.set_pipeline(self.trace_pipe)
                cp.set_bind_group(0, group)
                cp.set_bind_group(1, self.material_group)
                cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
                cp.end()
                d.queue.submit([enc.finish()])
                self.samples += 1
        self._show(target, look, linear)
        return self.samples

    def _show(self, target, look, linear):
        d = self.device
        fmt = target.format
        if fmt not in self._display:
            mod = d.create_shader_module(code=DISPLAY)
            self._display[fmt] = d.create_render_pipeline(
                layout="auto", vertex={"module": mod, "entry_point": "vs"},
                fragment={"module": mod, "entry_point": "fs", "targets": [{"format": fmt}]},
                primitive={"topology": "triangle-list"})
        pipe = self._display[fmt]
        d.queue.write_buffer(self.display_uniform, 0, struct.pack(
            "f 7I", look.exposure, look.tonemap, look.channel, int(linear), self.size[0], 0, 0, 0))
        group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": self.display_uniform}},
            {"binding": 1, "resource": {"buffer": self.accum}},
            {"binding": 2, "resource": self.r.ocio_lut.create_view()},
            {"binding": 3, "resource": self.r.clamp_sampler}])
        enc = d.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{"view": target.create_view(), "load_op": "clear",
                                                       "store_op": "store", "clear_value": (0, 0, 0, 1)}])
        rp.set_pipeline(pipe)
        rp.set_bind_group(0, group)
        rp.draw(3)
        rp.end()
        d.queue.submit([enc.finish()])
