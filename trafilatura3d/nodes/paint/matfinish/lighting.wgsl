// Baked Lighting: the shading normal in world space and the material read in linear values.
// Needs the inputs `normal`, `height` (the channels), `baked_normal`, `world_space_normals`, `position`.

// Tangent frame at a texel: the geometric normal from the world space normal map, the tangent and bitangent from
// how the position map changes along U and V (the gentler one-sided difference, so UV seams don't break it).
// The position map is normalized per axis to the mesh's box, which skews these directions on a mesh whose box
// is far from a cube. Without a position map (or where it is flat): a frame around the normal from world X.
fn bl_side(c: vec2i, d: vec2i) -> vec3f {
    let p0 = load_position(c).rgb;
    let a = load_position(c + d).rgb - p0;
    let b = p0 - load_position(c - d).rgb;
    return select(b, a, dot(a, a) < dot(b, b));
}
fn bl_frame(c: vec2i) -> mat3x3f {
    let n = normalize(load_world_space_normals(c).rgb * 2.0 - 1.0 + vec3f(0.0, 1e-5, 0.0));
    var t = vec3f(0.0); var b = vec3f(0.0);
    if (POSITION_CONNECTED) {
        let du = bl_side(c, vec2i(1, 0));
        let dv = -bl_side(c, vec2i(0, 1));  // rows grow downwards, V upwards
        t = du - n * dot(n, du);
        b = dv - n * dot(n, dv);
    }
    if (dot(t, t) < 1e-12 || dot(b, b) < 1e-12) {
        let ref_axis = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.x) > 0.99);
        t = ref_axis - n * dot(n, ref_axis);
        b = cross(n, normalize(t));
    }
    return mat3x3f(normalize(t), normalize(b), n);
}
fn bl_tangent(rgb: vec3f) -> vec3f { return normalize(rgb * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5)); }
fn bl_whiteout(a: vec3f, b: vec3f) -> vec3f { return normalize(vec3f(a.xy + b.xy, a.z * b.z)); }
// World space shading normal: the mesh (world space normals + the baked normal map) and, with `details`, the
// normal channel and the height channel's slopes x `height_k` (0..1).
fn bl_normal(c: vec2i, details: bool, height_k: f32) -> vec3f {
    var t = bl_tangent(load_baked_normal(c).rgb);
    if (details) {
        t = bl_whiteout(t, bl_tangent(load_normal(c).rgb));
        let sx = (load_height(c + vec2i(1, 0)).r - load_height(c - vec2i(1, 0)).r) * 0.5;
        let sy = (load_height(c - vec2i(0, 1)).r - load_height(c + vec2i(0, 1)).r) * 0.5;
        let k = height_k * 10.0 * p.size_.x / 256.0;
        t = bl_whiteout(t, normalize(vec3f(-sx * k, -sy * k, 1.0)));
    }
    return normalize(bl_frame(c) * t);
}
fn bl_lin(c: vec3f) -> vec3f { return srgb_to_linear(clamp(c, vec3f(0.0), vec3f(1.0))); }
fn bl_out(c: vec3f) -> vec3f { return clamp(linear_to_srgb(c), vec3f(0.0), vec3f(1.0)); }
// A light's direction from two angles (degrees): horizontal around +Y from +Z, vertical above the horizon.
fn bl_dir(h: f32, v: f32) -> vec3f {
    let a = radians(h); let e = radians(v);
    return normalize(vec3f(cos(e) * sin(a), sin(e), cos(e) * cos(a)));
}
