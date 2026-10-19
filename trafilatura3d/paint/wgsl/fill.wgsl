// A uniform value or an image through the fill's projection (Fill projections pages). The image coordinates are
// scaled by Tiling around the center, rotated, offset, repeated as UV Wrap says (outside: transparent) and
// filtered. Grayscale channels take the image's luminance.
// UV projection: the texel's UV. Planar: the mesh position (texture space map `position`) in the projection's
// space, its XY plane spanning [-0.5, 0.5] (scale 1 = the bounding box's largest side), projected along Z;
// Depth culling fades beyond |z| > 0.5, Backface culling fades the faces turned away from +Z by more than the
// angle (ours: the documentation gives neither formula). Tri-planar: the planar projection along X, Y and Z,
// weighted by the mesh normal raised to a power growing with Hardness (1 = clear cut). Shape crop "Project
// cropped to shape": nothing outside the projection's square (Planar) or cube (Tri-planar). Spherical: longitude
// and latitude around the center; Cylindrical: around the Y axis, the image's width over Angle of the perimeter,
// Backface culling fading faces that do not look away from the axis (ours: the pages give no formula).

fn texel_at(q: vec2i) -> vec4f { return texel_image(q); }

fn bilinear(uv: vec2f) -> vec4f {
    let s = vec2f(size_image());
    let x = uv * s - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    return mix(mix(texel_at(i), texel_at(i + vec2i(1, 0)), f.x), mix(texel_at(i + vec2i(0, 1)), texel_at(i + vec2i(1, 1)), f.x), f.y);
}

// The image at projection coordinates `uv` (0-1 = the projection's square) through the UV transformation;
// alpha 0 outside the image when UV Wrap does not repeat.
var<private> g_flip: i32 = 0;  // a symmetry copy's Flip copy: 1 mirrors the image in U, 2 in V

fn project(uv: vec2f) -> vec4f {
    let a = radians(p.rotation);
    let r = mat2x2f(cos(a), sin(a), -sin(a), cos(a));
    var q = r * (uv - 0.5 - p.offset) * p.tiling + 0.5;
    if (g_flip == 1) { q.x = 1.0 - q.x; }
    if (g_flip == 2) { q.y = 1.0 - q.y; }
    // UV Wrap: 0 None, 1 Repeat horizontally, 2 Repeat vertically, 3 Repeat.
    let out_u = (q.x < 0.0 || q.x > 1.0) && (p.wrap & 1) == 0;
    let out_v = (q.y < 0.0 || q.y > 1.0) && (p.wrap & 2) == 0;
    if (out_u || out_v) { return vec4f(0.0); }
    let ts = vec2f(size_image());
    if (p.filtering == 2) {  // Nearest
        return texel_at(vec2i(floor(q * ts)));
    }
    if (p.filtering == 0) {  // Bilinear HQ: the texel footprint averaged (2x2 taps)
        let fp = p.tiling / p.size_ * 0.25;
        return 0.25 * (bilinear(q + r * vec2f(-fp.x, -fp.y)) + bilinear(q + r * vec2f(fp.x, -fp.y))
                     + bilinear(q + r * vec2f(-fp.x, fp.y)) + bilinear(q + r * vec2f(fp.x, fp.y)));
    }
    return bilinear(q);  // Bilinear Sharp
}

fn in_square(l: vec2f) -> bool { return all(abs(l) <= vec2f(0.5)); }

// The Warp surface at parameters uv (0-1, v from the top): the sampled grid bilinearly, extended straight beyond
// its borders.
fn warp_at(uv: vec2f) -> vec3f {
    let sz = vec2f(size_warp());
    let uc = clamp(uv, vec2f(0.0), vec2f(1.0));
    let x = uc * (sz - 1.0);
    let i = vec2i(min(floor(x), sz - 2.0));
    let f = x - vec2f(i);
    let a = texel_warp(i).xyz;
    let b = texel_warp(i + vec2i(1, 0)).xyz;
    let c = texel_warp(i + vec2i(0, 1)).xyz;
    let d = texel_warp(i + vec2i(1, 1)).xyz;
    let s = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    let du = (mix(b, d, f.y) - mix(a, c, f.y)) * (sz.x - 1.0);
    let dv = (mix(c, d, f.x) - mix(a, b, f.x)) * (sz.y - 1.0);
    return s + du * (uv.x - uc.x) + dv * (uv.y - uc.y);
}

// Warp: the image coordinates of the surface point nearest to `l` (Gauss-Newton from the flat guess) and the
// signed distance along the surface's normal.
fn warp_project(l: vec3f) -> vec3f {
    var uv = vec2f(l.x + 0.5, 0.5 - l.y);
    let h = 1.0 / 128.0;
    var dist = 0.0;
    for (var k = 0; k < 8; k++) {
        let s = warp_at(uv);
        let su = (warp_at(uv + vec2f(h, 0.0)) - warp_at(uv - vec2f(h, 0.0))) / (2.0 * h);
        let sv = (warp_at(uv + vec2f(0.0, h)) - warp_at(uv - vec2f(0.0, h))) / (2.0 * h);
        let r = l - s;
        let a = dot(su, su); let b = dot(su, sv); let c = dot(sv, sv);
        let det = a * c - b * b;
        if (abs(det) < 1e-12) { break; }
        let ru = dot(su, r); let rv = dot(sv, r);
        uv += vec2f(c * ru - b * rv, a * rv - b * ru) / det;
        uv = clamp(uv, vec2f(-1.0), vec2f(2.0));
        dist = dot(r, normalize(cross(sv, su)));
    }
    return vec3f(uv, dist);
}

// The image seen through the projection at texel c (alpha = how much of it reaches the texel).
fn projected(c: vec2i) -> vec4f {
    if (p.mode == 0) { return project((vec2f(c) + 0.5) / p.size_); }
    let world = load_position(c).xyz;
    let wnormal = load_geonormal(c).xyz;
    var v = projected_at(world, wnormal);
    // Symmetry (fill layers with a 3D projection): every copy's content where it lands, over the original. A copy
    // maps x to M x + t: the texel at `world` shows the projection at M^T (world - t). Texture `sym`: per copy
    // three texels the rows of M^T with -(M^T t), then one with the Flip copy.
    if (SYM_CONNECTED) {
        let count = i32(size_sym().x) / 4;
        for (var k = 0; k < count; k++) {
            let r0 = texel_sym(vec2i(4 * k, 0));
            let r1 = texel_sym(vec2i(4 * k + 1, 0));
            let r2 = texel_sym(vec2i(4 * k + 2, 0));
            let x = vec4f(world, 1.0);
            let src = vec3f(dot(r0, x), dot(r1, x), dot(r2, x));
            let nsrc = vec3f(dot(r0.xyz, wnormal), dot(r1.xyz, wnormal), dot(r2.xyz, wnormal));
            g_flip = i32(texel_sym(vec2i(4 * k + 3, 0)).x + 0.5);
            let t = projected_at(src, nsrc);
            g_flip = 0;
            v = vec4f(mix(v.rgb, t.rgb, t.a), t.a + v.a * (1.0 - t.a));
        }
    }
    return v;
}

// The projection at a world position and normal (3D projections).
fn projected_at(world: vec3f, wn: vec3f) -> vec4f {
    let wp = vec4f(world, 1.0);
    let l = vec3f(dot(p.m0, wp), dot(p.m1, wp), dot(p.m2, wp));  // projection space
    let n = normalize(vec3f(dot(p.m0.xyz, wn), dot(p.m1.xyz, wn), dot(p.m2.xyz, wn)) + vec3f(1e-6));
    if (p.mode == 5) {  // Warp: through the grid's surface, within Projection depth of it
        let q = warp_project(l);
        var t = project(q.xy);
        var w = 1.0 - smoothstep(p.warp_depth * p.depth_hardness, p.warp_depth + 1e-5, abs(q.z));
        if (p.crop != 0 && (any(q.xy < vec2f(0.0)) || any(q.xy > vec2f(1.0)))) { w = 0.0; }
        return vec4f(t.rgb, t.a * w);
    }
    if (p.mode == 3) {  // Spherical: longitude and latitude around the projection's center (Y up)
        let d = normalize(l + vec3f(0.0, 1e-9, 0.0));
        var t = project(vec2f(atan2(d.x, d.z) / 6.2831853 + 0.5, acos(clamp(d.y, -1.0, 1.0)) / 3.14159265));
        if (p.crop != 0 && length(l) > 0.5) { t.a = 0.0; }
        return t;
    }
    if (p.mode == 4) {  // Cylindrical: around the projection's Y axis, Angle of the perimeter for the image's width
        let th = atan2(l.x, l.z);
        var t = project(vec2f(th / p.angle + 0.5, 0.5 - l.y));
        var w = 1.0;
        if (p.crop != 0 && (length(l.xz) > 0.5 || abs(l.y) > 0.5 || abs(th) > p.angle * 0.5)) { w = 0.0; }
        if (p.backface_culling != 0) {  // faces turned away from the axis (perpendicular to the cylinder) fade
            let radial = normalize(vec3f(l.x, 0.0, l.z) + vec3f(1e-9, 0.0, 0.0));
            let soft = (1.0 - p.backface_hardness) * 0.5 + 1e-4;
            w *= smoothstep(0.0, soft, dot(n, radial));
        }
        return vec4f(t.rgb, t.a * w);
    }
    if (p.mode == 2) {  // Planar, along Z
        var t = project(vec2f(l.x, -l.y) + 0.5);
        var w = 1.0;
        if (p.crop != 0 && !in_square(l.xy)) { w = 0.0; }
        if (p.depth_culling != 0) {
            let soft = (1.0 - p.depth_hardness) * 0.5 + 1e-4;
            w *= 1.0 - smoothstep(0.5, 0.5 + soft, abs(l.z));
        }
        if (p.backface_culling != 0) {
            let soft = (1.0 - p.backface_hardness) * 0.5 + 1e-4;
            w *= smoothstep(p.backface_cos - soft, p.backface_cos, n.z);
        }
        return vec4f(t.rgb, t.a * w);
    }
    // Tri-planar
    let k = mix(1.0, 32.0, p.hardness);
    var bw = pow(abs(n), vec3f(k));
    if (p.hardness >= 1.0) {  // clear cut: the dominant axis only
        let m = max(abs(n.x), max(abs(n.y), abs(n.z)));
        bw = select(vec3f(0.0), vec3f(1.0), abs(n) >= vec3f(m));
    }
    bw /= max(bw.x + bw.y + bw.z, 1e-6);
    let tx = project(vec2f(select(-l.z, l.z, n.x < 0.0), -l.y) + 0.5);
    let ty = project(vec2f(l.x, select(-l.z, l.z, n.y < 0.0)) + 0.5);
    let tz = project(vec2f(select(l.x, -l.x, n.z < 0.0), -l.y) + 0.5);
    let t = tx * bw.x + ty * bw.y + tz * bw.z;
    let inside = p.crop == 0 || all(abs(l) <= vec3f(0.5));
    return select(vec4f(t.rgb, 0.0), t, inside);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    var v = vec4f(p.color.rgb, p.alpha);
    if (IMAGE_CONNECTED) {
        var t = projected(c);
        if (t.a <= 0.0) {
            v = vec4f(p.color.rgb, 0.0);
        } else {
            if (p.flip_green != 0) { t.g = 1.0 - t.g; }  // a material normal in the other format
            if (p.signed_ != 0) { t = vec4f(t.rgb * 2.0 - 1.0, t.a); }  // Height: [0, 1] sources to [-1, 1]
            v = vec4f(t.rgb, p.alpha * t.a);
            if (p.gray != 0) { v = vec4f(vec3f(luminance(t.rgb)), v.a); }
        }
    }
    store(c, v);
}
