"""The bakers' WGSL (see bake.py): the low poly's G-buffer in UV space, the primary pass (one ray from the low
surface to the high poly per sample, and the bakers that need nothing more), the secondary pass (occlusion,
bent normals, thickness and curvature rays from the primary hits, against the BVH Self Occlusion picks) and
dilation."""

# --- G-buffer: the low poly in UV space ---------------------------------------------------------------------
# Targets: world position, ray direction + match group, normal + material, tangent, cage + vertex curvature.
GBUFFER = """
@group(0) @binding(0) var<uniform> view: vec4f;  // UV -> clip space of the tile: scale xy, offset zw
struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f, @location(1) ray: vec3f, @location(2) normal: vec3f,
    @location(3) tangent: vec4f, @location(4) cage: vec3f, @location(5) @interpolate(flat) group: f32,
    @location(6) curv: f32, @location(7) @interpolate(flat) material: f32,
};
@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) t: vec4f, @location(3) uv: vec2f,
      @location(4) ray: vec3f, @location(5) cage: vec3f, @location(6) group: f32, @location(7) curv: f32,
      @builtin(instance_index) material: u32) -> VOut {
    var o: VOut;
    o.pos = vec4f(uv * view.xy + view.zw, 0.0, 1.0);  // UV v grows down the image
    o.world = p; o.ray = ray; o.normal = n; o.tangent = t; o.cage = cage; o.group = group; o.curv = curv;
    o.material = f32(material);  // drawn per submesh with its material as the instance
    return o;
}
// Two rgba32float targets per pass (the default limit is 32 bytes of color attachments per sample).
struct F2 { @location(0) a: vec4f, @location(1) b: vec4f };
@fragment
fn fs0(i: VOut) -> F2 { return F2(vec4f(i.world, 1.0), vec4f(normalize(i.ray), i.group)); }
@fragment
fn fs1(i: VOut) -> F2 { return F2(vec4f(normalize(i.normal), i.material), i.tangent); }
@fragment
fn fs2(i: VOut) -> @location(0) vec4f { return vec4f(i.cage, i.curv); }
"""

# Shared by both passes: random numbers, ray cones, the low frame, normal map decoding.
COMMON = """
fn basis(n: vec3f) -> mat3x3f {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.999);
    let t = normalize(cross(up, n));
    return mat3x3f(t, cross(n, t), n);
}
// The texel's rotation of the ray pattern (Cranley-Patterson), from the blue noise mask.
var<private> rot: vec2f;
// Ray i of count in the cone of half angle acos(cos_max) around n, cosine weighted or uniform in solid angle:
// the Hammersley point set, rotated by the texel's `rot`, so the rays of a texel cover the cone evenly.
fn cone(n: vec3f, cos_max: f32, cosine: bool, i: u32, count: u32) -> vec3f {
    let u = fract((f32(i) + 0.5) / f32(count) + rot.x);
    let phi = 6.2831853 * fract(f32(reverseBits(i)) / 4294967296.0 + rot.y);
    var z: f32;
    if (cosine) {
        let s = sqrt(u) * sqrt(max(1.0 - cos_max * cos_max, 0.0));
        z = sqrt(max(1.0 - s * s, 0.0));
    } else {
        z = mix(1.0, cos_max, u);
    }
    let r = sqrt(max(1.0 - z * z, 0.0));
    return normalize(basis(n) * vec3f(r * cos(phi), r * sin(phi), z));
}
// Tangent space (OpenGL: green up the image, i.e. against the bitangent bv that follows the rows) <-> world.
fn to_tangent(v: vec3f, tg: vec3f, bv: vec3f, n: vec3f, directx: bool) -> vec3f {
    var t = vec3f(dot(v, tg), -dot(v, bv), dot(v, n));
    if (directx) { t.y = -t.y; }
    return t;
}
fn from_tangent(t0: vec3f, tg: vec3f, bv: vec3f, n: vec3f, directx: bool) -> vec3f {
    var t = t0;
    if (directx) { t.y = -t.y; }
    return normalize(tg * t.x - bv * t.y + n * t.z);
}
fn hash_color(id: u32) -> vec3f {
    var s = id * 2654435761u + 12345u;
    s = (s ^ (s >> 16u)) * 2246822519u;
    s = s ^ (s >> 13u);
    return vec3f(f32(s & 255u), f32((s >> 8u) & 255u), f32((s >> 16u) & 255u)) / 255.0;
}
fn hue(h: f32) -> vec3f {
    return clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}
"""

# --- primary pass --------------------------------------------------------------------------------------------
# Modes: 1 normal map from mesh, 5 height (raw signed distance), 6 position map from mesh, 7 color map, 8
# transferred texture, 9 opacity, 10 position, 11 world space direction, 14 world space normals, 16-20 the low's
# raw normal / tangent / position / bitangent / vertex curvature (Curvature), 30 the hits for the secondary pass
# (aux: hit position + status, base normal + group), 31 the same from the low surface itself (Ambient Occlusion).
PRIMARY = COMMON + """
struct P {
    mode: u32, group: u32, use_cage: u32, leaf_base: u32,
    tri_count: u32, o0: u32, o1: u32, o2: u32,
    o3: u32, nmat: u32, any_group: u32, skew: u32,  // skew: 0 off, 1 Skew Map, 2 inverted
    frontal: f32, rear: f32, f0: f32, f1: f32,
    v0: vec4f,
    origin: vec4u,  // the tile's first sample in the map, the UDIM tile index
    dims: vec4u,    // the map's size in samples; z: the rays ignore backface hits (Ignore backface)
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var g_world: texture_2d<f32>;
@group(0) @binding(2) var g_ray: texture_2d<f32>;
@group(0) @binding(3) var g_normal: texture_2d<f32>;
@group(0) @binding(4) var g_tangent: texture_2d<f32>;
@group(0) @binding(5) var g_cage: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> verts: array<vec4f>;  // high: position, normal, tangent per vertex
@group(0) @binding(7) var<storage, read> tris: array<vec4u>;
@group(0) @binding(8) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(9) var<storage, read> attrs: array<vec4f>;  // high, 3 per vertex: (uv, mesh, material),
                                                               // vertex color, (material color, element)
@group(0) @binding(10) var<storage, read_write> out: array<vec4f>;
@group(0) @binding(11) var tex: texture_2d<f32>;  // the baker's input texture
@group(0) @binding(12) var samp: sampler;
@group(0) @binding(13) var<storage, read_write> out2: array<vec4f>;
@group(0) @binding(14) var<storage, read> boxes: array<vec4f>;  // per material min, max; the whole at nmat
@group(0) @binding(15) var skew_map: texture_2d<f32>;

var<private> bvh_leaf_base: u32;
var<private> bvh_tri_count: u32;

fn hit_attr(h: BvhHit, k: u32) -> vec4f {
    let t = tris[h.tri];
    let w = 1.0 - h.b.x - h.b.y;
    return attrs[t.x * 3u + k] * w + attrs[t.y * 3u + k] * h.b.x + attrs[t.z * 3u + k] * h.b.y;
}
fn hit_vert(h: BvhHit, k: u32) -> vec4f {
    let t = tris[h.tri];
    let w = 1.0 - h.b.x - h.b.y;
    return verts[t.x * 3u + k] * w + verts[t.y * 3u + k] * h.b.x + verts[t.z * 3u + k] * h.b.y;
}
// Position bakers: per axis in the box (BBox) or in the bounding sphere (BSphere), all axes or one.
fn position(q: vec3f, box: u32) -> vec4f {
    let lo = boxes[box * 2u].xyz;
    let hi = boxes[box * 2u + 1u].xyz;
    var v: vec3f;
    if (p.o1 == 1u) {
        let r = max(length(hi - lo) * 0.5, 1e-9);
        v = (q - (lo + hi) * 0.5) / (2.0 * r) + 0.5;
    } else {
        v = (q - lo) / max(hi - lo, vec3f(1e-9));
    }
    if (p.o0 > 0u) { let a = v[p.o0 - 1u]; return vec4f(a, a, a, 1.0); }
    return vec4f(v, 1.0);
}
fn tex_at(uv: vec2f, nearest: bool) -> vec4f {
    if (nearest) {
        let size = vec2i(textureDimensions(tex));
        let c = clamp(vec2i(fract(uv) * vec2f(size)), vec2i(0), size - 1);
        return textureLoad(tex, c, 0);
    }
    return textureSampleLevel(tex, samp, uv, 0.0);
}
fn id_color(id: f32, count: f32) -> vec3f {
    switch p.o1 {
        case 1u: { return hue(id / max(count, 1.0)); }
        case 2u: { let g = (id + 1.0) / max(count, 1.0); return vec3f(g); }
        default: { return hash_color(u32(id)); }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(g_world);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let c = vec2i(id.xy);
    let pix = id.y * size.x + id.x;
    let world = textureLoad(g_world, c, 0);
    let ray = textureLoad(g_ray, c, 0);
    if (world.w < 0.5 || (p.any_group == 0u && u32(ray.w + 0.5) != p.group)) { return; }
    bvh_leaf_base = p.leaf_base;
    bvh_tri_count = p.tri_count;
    bvh_cull = p.dims.z;
    let gx = p.origin.x + id.x;
    let gy = p.origin.y + id.y;
    let uv = (vec2f(f32(gx), f32(gy)) + 0.5) / vec2f(p.dims.xy);  // in the UDIM tile
    let nw = textureLoad(g_normal, c, 0);
    let n_low = normalize(nw.xyz);
    let material = u32(nw.w + 0.5);
    let tan4 = textureLoad(g_tangent, c, 0);
    let tg = normalize(tan4.xyz - n_low * dot(n_low, tan4.xyz));
    let bv = cross(n_low, tg) * select(-1.0, 1.0, tan4.w >= 0.0);  // follows the image rows
    let pos = world.xyz;
    let box = select(p.nmat, material, p.o2 == 1u);

    // The bakers of the low mesh alone.
    switch p.mode {
        case 10u: { out[pix] = position(pos, min(box, p.nmat)); return; }
        case 11u: {  // world space direction, expressed in the low's tangent space
            var d = normalize(p.v0.xyz);
            if (p.o0 == 1u) { d = normalize(tex_at(uv, false).xyz * 2.0 - 1.0); }
            out[pix] = vec4f(to_tangent(d, tg, bv, n_low, p.o1 == 1u) * 0.5 + 0.5, 1.0);
            return;
        }
        case 14u: {  // world space normals / tangents / binormals
            var v = n_low;
            if (p.o0 == 1u) { v = tg; } else if (p.o0 == 2u) { v = bv; }
            else if (p.o1 == 1u) { v = from_tangent(tex_at(uv, false).xyz * 2.0 - 1.0, tg, bv, n_low, p.o2 == 1u); }
            out[pix] = vec4f(v * 0.5 + 0.5, 1.0);
            return;
        }
        case 16u: { out[pix] = vec4f(n_low, 1.0); return; }
        case 17u: { out[pix] = vec4f(tg, 1.0); return; }
        case 18u: { out[pix] = vec4f(pos, 1.0); return; }
        case 19u: { out[pix] = vec4f(bv, 1.0); return; }
        case 20u: { let k = textureLoad(g_cage, c, 0).w; out[pix] = vec4f(k, k, k, 1.0); return; }
        case 31u: {  // occlusion from the low surface itself (Ambient Occlusion): the base normal
            var nb = n_low;
            if (p.o0 == 1u) {
                let t = tex_at(uv, false).xyz * 2.0 - 1.0;
                nb = select(from_tangent(t, tg, bv, n_low, p.o2 == 1u), normalize(t), p.o1 == 1u);
            }
            if (p.o3 == 1u) { nb = -nb; }  // Invert Normal
            out[pix] = vec4f(pos, 2.0);
            out2[pix] = vec4f(nb, ray.w);
            return;
        }
        default: {}
    }

    // Ray from the low surface to the high meshes: within [-rear, +frontal] along the ray direction, or from
    // the cage towards the low surface and as far beyond.
    var o: vec3f;
    var d: vec3f;
    var tmax: f32;
    if (p.use_cage == 1u) {
        let cage = textureLoad(g_cage, c, 0).xyz;
        let span = pos - cage;
        tmax = length(span) * 2.0;
        d = span / max(length(span), 1e-9);
        o = cage;
    } else {
        var dir = normalize(ray.xyz);
        if (p.skew > 0u) {  // Skew Correction: black the averaged normal, white the original one
            var k = textureSampleLevel(skew_map, samp, uv, 0.0).r;
            if (p.skew == 2u) { k = 1.0 - k; }
            dir = normalize(mix(dir, n_low, clamp(k, 0.0, 1.0)));
        }
        d = -dir;
        o = pos - d * p.frontal;
        tmax = p.frontal + p.rear;
    }
    let h = bvh_trace(o, d, 0.0, tmax, false);
    let hit = h.tri != 0xffffffffu;
    if (p.mode == 9u) { let v = select(0.0, 1.0, hit); out[pix] = vec4f(v, v, v, 1.0); return; }
    if (!hit) {
        // A missed ray inside a UV island: the neutral value (what the low surface gives by itself), not a hole
        // for dilation to smear across the island.
        switch p.mode {
            case 1u: { out[pix] = select(vec4f(0.5, 0.5, 1.0, 1.0), vec4f(n_low * 0.5 + 0.5, 1.0), p.o0 == 1u); }
            case 5u: { out[pix] = vec4f(0.0, 0.0, 0.0, 1.0); }  // on the low surface
            case 6u: { out[pix] = position(pos, min(box, p.nmat)); }
            case 30u: { out[pix] = vec4f(pos, 1.0); out2[pix] = vec4f(n_low, ray.w); }
            default: { out[pix] = vec4f(0.0, 0.0, 0.0, 1.0); }
        }
        return;
    }
    let hp = o + d * h.t;
    let nh = normalize(hit_vert(h, 1u).xyz);
    let first = tris[h.tri].x;
    switch p.mode {
        case 1u: {
            if (p.o0 == 1u) { out[pix] = vec4f(nh * 0.5 + 0.5, 1.0); return; }
            out[pix] = vec4f(normalize(to_tangent(nh, tg, bv, n_low, p.o1 == 1u)) * 0.5 + 0.5, 1.0);
        }
        case 5u: { let dist = dot(hp - pos, normalize(ray.xyz)); out[pix] = vec4f(dist, dist, dist, 1.0); }
        case 6u: {
            let hb = select(p.nmat, u32(attrs[first * 3u].w + 0.5), p.o2 == 1u);
            out[pix] = position(hp, min(hb, p.nmat));
        }
        case 7u: {
            var col: vec3f;
            switch p.o0 {
                case 0u: { col = hit_attr(h, 1u).rgb; }  // vertex color, interpolated
                case 1u: { col = attrs[first * 3u + 2u].rgb; }  // material color, per face
                case 2u: { col = id_color(attrs[first * 3u].z, p.v0.x); }  // mesh ID
                default: { col = id_color(attrs[first * 3u + 2u].w, p.v0.y); }  // element (polygroup / submesh)
            }
            out[pix] = vec4f(col, 1.0);
        }
        case 8u: {  // transferred texture at the hit's UV; a normal map is re-expressed in the low's frame
            let a = hit_attr(h, 0u);
            var v = tex_at(a.xy, p.o0 == 1u);
            if (p.o1 == 1u && p.o2 == 0u) {
                let ht4 = hit_vert(h, 2u);
                let ht = normalize(ht4.xyz - nh * dot(nh, ht4.xyz));
                let hb = cross(nh, ht) * select(-1.0, 1.0, ht4.w >= 0.0);
                let wn = from_tangent(v.xyz * 2.0 - 1.0, ht, hb, nh, p.o3 == 1u);
                v = vec4f(normalize(to_tangent(wn, tg, bv, n_low, p.o3 == 1u)) * 0.5 + 0.5, 1.0);
            }
            out[pix] = vec4f(v.rgb, 1.0);
        }
        case 30u: {  // the hit for the secondary rays, with its base normal (the hit's, or a normal map's)
            var nb = nh;
            if (p.o0 == 1u) {
                let t = tex_at(uv, false).xyz * 2.0 - 1.0;
                nb = select(from_tangent(t, tg, bv, n_low, p.o2 == 1u), normalize(t), p.o1 == 1u);
            }
            out[pix] = vec4f(hp, 2.0);
            out2[pix] = vec4f(nb, ray.w);
        }
        default: {}
    }
}
"""

# --- secondary pass ------------------------------------------------------------------------------------------
# From the primary hits (aux: position + status 0 empty / 1 missed / 2 hit, base normal + group). Modes: 2
# occlusion (Ambient Occlusion from Mesh, and Ambient Occlusion from the low), 3 bent normals, 4 thickness, 12
# curvature from mesh (raw: the mean relative difference between the rays' hit distances and a flat surface's).
SECONDARY = COMMON + """
struct S {
    mode: u32, rays: u32, cosine: u32, leaf_base: u32,
    tri_count: u32, cull: u32, attenuation: u32, ground: u32,
    group: u32, any_group: u32, world: u32, directx: u32,
    min_occ: f32, max_occ: f32, cos_spread: f32, ground_y: f32,
    radius: f32, pad0: f32, pad1: f32, pad2: f32,
    origin: vec4u,
    dims: vec4u,
};
@group(0) @binding(0) var<uniform> s: S;
@group(0) @binding(1) var<storage, read> aux: array<vec4f>;
@group(0) @binding(2) var<storage, read> aux2: array<vec4f>;
@group(0) @binding(3) var<storage, read> verts: array<vec4f>;
@group(0) @binding(4) var<storage, read> tris: array<vec4u>;
@group(0) @binding(5) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> out: array<vec4f>;
@group(0) @binding(7) var g_normal: texture_2d<f32>;
@group(0) @binding(8) var g_tangent: texture_2d<f32>;
@group(0) @binding(9) var<storage, read> blue: array<vec2f>;  // bake.blue_noise, tiled on the map

var<private> bvh_leaf_base: u32;
var<private> bvh_tri_count: u32;

// The nearest occluder along a ray: the BVH, and the ground plane (an XZ plane under the bounding box).
fn occluder(o: vec3f, d: vec3f, tmin: f32, tmax: f32) -> f32 {
    let h = bvh_trace(o, d, tmin, tmax, false);
    var t = select(1e30, h.t, h.tri != 0xffffffffu);
    if (s.ground == 1u && d.y < -1e-6) {
        let tg = (s.ground_y - o.y) / d.y;
        if (tg > tmin && tg < tmax) { t = min(t, tg); }
    }
    return t;
}
fn attenuate(t: f32) -> f32 {
    let x = clamp(t / s.max_occ, 0.0, 1.0);
    switch s.attenuation {
        case 0u: { return 1.0; }
        case 2u: { return 1.0 - smoothstep(0.0, 1.0, x); }
        default: { return 1.0 - x; }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(g_normal);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let pix = id.y * size.x + id.x;
    let a = aux[pix];
    let b = aux2[pix];
    if (a.w < 0.5 || (s.any_group == 0u && u32(b.w + 0.5) != s.group)) { return; }
    bvh_leaf_base = s.leaf_base;
    bvh_tri_count = s.tri_count;
    bvh_cull = s.cull;
    let gx = s.origin.x + id.x;
    let gy = s.origin.y + id.y;
    // Neighbouring texels get far apart rotations, so their errors differ: blue noise instead of blotches. The
    // height of the rays only moves within one stratum (1 / rays): a whole-range shift reshuffles the
    // pattern, and close rotations would no longer give close errors.
    let side = u32(round(sqrt(f32(arrayLength(&blue)))));
    rot = blue[(gy % side) * side + gx % side];
    rot.x /= f32(max(s.rays, 1u));
    let c = vec2i(id.xy);
    let n_low = normalize(textureLoad(g_normal, c, 0).xyz);
    let tan4 = textureLoad(g_tangent, c, 0);
    let tg = normalize(tan4.xyz - n_low * dot(n_low, tan4.xyz));
    let bv = cross(n_low, tg) * select(-1.0, 1.0, tan4.w >= 0.0);
    let hp = a.xyz;
    let nb = normalize(b.xyz);
    let missed = a.w < 1.5;
    let rays = max(s.rays, 1u);
    switch s.mode {
        case 2u: {  // occlusion: 1 - the attenuated share of rays meeting an occluder
            if (missed) { out[pix] = vec4f(1.0, 1.0, 1.0, 1.0); return; }
            var occ = 0.0;
            let o = hp + nb * max(s.min_occ * 4.0, 1e-6);
            for (var i = 0u; i < rays; i++) {
                let t = occluder(o, cone(nb, s.cos_spread, s.cosine == 1u, i, rays), s.min_occ, s.max_occ);
                if (t < s.max_occ) { occ += attenuate(t); }
            }
            let v = 1.0 - occ / f32(rays);
            out[pix] = vec4f(v, v, v, 1.0);
        }
        case 3u: {  // bent normals: the mean free direction
            var bent = vec3f(0.0);
            if (!missed) {
                let o = hp + nb * max(s.min_occ * 4.0, 1e-6);
                for (var i = 0u; i < rays; i++) {
                    let r = cone(nb, s.cos_spread, s.cosine == 1u, i, rays);
                    if (occluder(o, r, s.min_occ, s.max_occ) >= s.max_occ) { bent += r; }
                }
            }
            let v = select(nb, normalize(bent), length(bent) > 1e-6);
            if (s.world == 1u) { out[pix] = vec4f(v * 0.5 + 0.5, 1.0); return; }
            out[pix] = vec4f(normalize(to_tangent(v, tg, bv, n_low, s.directx == 1u)) * 0.5 + 0.5, 1.0);
        }
        case 4u: {  // thickness: rays into the surface, their mean hit distance over the max distance
            if (missed) { out[pix] = vec4f(1.0, 1.0, 1.0, 1.0); return; }
            var acc = 0.0;
            let o = hp - nb * max(s.min_occ * 4.0, 1e-6);
            for (var i = 0u; i < rays; i++) {
                let t = occluder(o, cone(-nb, s.cos_spread, s.cosine == 1u, i, rays), s.min_occ, s.max_occ);
                acc += select(1.0, t / s.max_occ, t < s.max_occ);
            }
            let v = acc / f32(rays);
            out[pix] = vec4f(v, v, v, 1.0);
        }
        case 12u: {  // curvature from mesh: rays from above the hit towards the surface around it
            if (missed) { out[pix] = vec4f(0.0, 0.0, 0.0, 1.0); return; }
            let o = hp + nb * s.radius;
            var k = 0.0;
            for (var i = 0u; i < rays; i++) {
                let r = cone(-nb, 0.5, false, i, rays);  // within 60 degrees of the surface's normal
                let plane = s.radius / max(dot(r, -nb), 0.1);  // where a flat surface would be
                let t = min(occluder(o, r, 0.0, plane * 3.0), plane * 3.0);
                k += (t - plane) / plane;  // farther: the surface falls away (convex); nearer: concave
            }
            let v = k / f32(rays);
            out[pix] = vec4f(v, v, v, 1.0);
        }
        default: {}
    }
}
"""

# --- denoise -------------------------------------------------------------------------------------------------
# Ours (the bakers' Denoise): an edge-aware a-trous filter (SVGF without the temporal part) on the secondary
# pass's output, guided by the primary hits. A neighbour counts when it is a hit, faces the same way, lies on the
# texel's tangent plane, is as far in 3D as its distance in the map says (not across a UV seam or another
# island), and differs in value no more than the noise explains (a real detail is kept). Step 0 estimates the
# noise (the variance among the 3x3 neighbours); steps 1, 2, 4, 8 filter (5x5 B-spline taps `step` samples
# apart), the variance with them.
DENOISE = """
struct D { size: vec2u, step: u32, pad: u32 };
@group(0) @binding(0) var<uniform> u: D;
@group(0) @binding(1) var<storage, read> aux: array<vec4f>;
@group(0) @binding(2) var<storage, read> aux2: array<vec4f>;
@group(0) @binding(3) var<storage, read> src: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(5) var<storage, read> var_src: array<f32>;
@group(0) @binding(6) var<storage, read_write> var_dst: array<f32>;

fn lum(v: vec4f) -> f32 { return (v.r + v.g + v.b) / 3.0; }

fn hit(c: vec2i) -> bool {
    return all(c >= vec2i(0)) && all(c < vec2i(u.size)) && aux[u32(c.y) * u.size.x + u32(c.x)].w > 1.5 &&
        src[u32(c.y) * u.size.x + u32(c.x)].w > 0.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= u.size.x || id.y >= u.size.y) { return; }
    let c = vec2i(id.xy);
    let pix = id.y * u.size.x + id.x;
    let v = src[pix];
    let p = aux[pix].xyz;
    let n = normalize(aux2[pix].xyz);
    if (u.step == 0u) {
        var m = vec2f(0.0);
        var k = 0.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let q = c + vec2i(dx, dy);
                if (!hit(q)) { continue; }
                let qi = u32(q.y) * u.size.x + u32(q.x);
                if (dot(n, normalize(aux2[qi].xyz)) < 0.9) { continue; }
                let l = lum(src[qi]);
                m += vec2f(l, l * l);
                k += 1.0;
            }
        }
        m /= max(k, 1.0);
        var_dst[pix] = max(m.y - m.x * m.x, 0.0);
        return;
    }
    dst[pix] = v;
    var_dst[pix] = var_src[pix];
    if (!hit(c)) { return; }
    let sigma = 4.0 * sqrt(var_src[pix]) + 1e-4;
    // The 3D size of one sample here: the nearest hit neighbour.
    var h = 1e30;
    for (var k = 0; k < 4; k++) {
        let q = c + array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1))[k];
        if (hit(q)) { h = min(h, length(aux[u32(q.y) * u.size.x + u32(q.x)].xyz - p)); }
    }
    if (h >= 1e30 || h <= 0.0) { return; }
    let kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    let step = f32(u.step);
    var acc = vec3f(0.0);
    var vacc = 0.0;
    var wsum = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = c + vec2i(dx, dy) * i32(u.step);
            if (!hit(q)) { continue; }
            let qi = u32(q.y) * u.size.x + u32(q.x);
            let dp = aux[qi].xyz - p;
            let reach = h * step * length(vec2f(f32(dx), f32(dy)));
            if (length(dp) > 3.0 * reach + h) { continue; }
            let off = dot(n, dp) / (h * step);
            let w = kernel[dx + 2] * kernel[dy + 2] * pow(max(dot(n, normalize(aux2[qi].xyz)), 0.0), 32.0) *
                exp(-off * off - abs(lum(src[qi]) - lum(v)) / sigma);
            acc += src[qi].rgb * w;
            vacc += var_src[qi] * w * w;
            wsum += w;
        }
    }
    dst[pix] = vec4f(acc / wsum, v.w);
    var_dst[pix] = vacc / (wsum * wsum);
}
"""

# Dilation on textures (no buffer size limit): an empty texel takes the mean of its filled neighbors.
DILATE = """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2i(textureDimensions(src));
    let c = vec2i(id.xy);
    if (c.x >= size.x || c.y >= size.y) { return; }
    let v = textureLoad(src, c, 0);
    if (v.a > 0.0) { textureStore(dst, c, v); return; }
    var acc = vec3f(0.0);
    var n = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = textureLoad(src, (c + vec2i(dx, dy) + size) % size, 0);
            if (s.a > 0.0) { acc += s.rgb; n += 1.0; }
        }
    }
    textureStore(dst, c, select(v, vec4f(acc / max(n, 1.0), 1.0), n > 0.0));
}
"""
