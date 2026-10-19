// Reoriented normal mapping: detail `d` laid on base `b` (unit vectors, Z up).
fn rnm(b: vec3f, d: vec3f) -> vec3f {
    let t = b + vec3f(0.0, 0.0, 1.0);
    let u = d * vec3f(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) / t.z - u);
}

fn effect(c: vec2i) -> vec4f {
    // Slopes per UV; image rows grow downwards, +Y (OpenGL green) is up.
    let g = vec2f(load_height(c + vec2i(1, 0)).r - load_height(c - vec2i(1, 0)).r,
                  load_height(c + vec2i(0, 1)).r - load_height(c - vec2i(0, 1)).r) * p.size_ * 0.5;
    var k = p.normal_intensity / 20.0;
    if (p.use_world_units != 0) { k = p.height_depth_cm / max(p.surface_size_cm, 1e-4); }
    var n = normalize(vec3f(-g.x * k, g.y * k, 1.0));
    if (p.overwrite_existing_normal == 0) { n = rnm(normalize(load_normal(c).xyz * 2.0 - 1.0), n); }
    return vec4f(clamp(n * 0.5 + 0.5, vec3f(0.0), vec3f(1.0)), 1.0);
}
