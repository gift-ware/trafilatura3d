// One 2x tiled copy of the noise: q = M (uv - 0.5) + 0.5 + offset, M twice an orthogonal matrix.
fn copy_at(uv: vec2f, m: vec4f, off: vec2f) -> f32 {
    let v = (uv - 0.5) * 2.0;
    return sample_grayscale(fract(vec2f(m.x * v.x + m.y * v.y, m.z * v.x + m.w * v.y) + 0.5 + off)).r;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    // Four copies, as is and turned by a quarter, a half and three quarters of a turn, each moved by its own
    // offset; the Mask picks within two pairs, the Mask turned by a quarter of a turn between the pairs.
    let a = copy_at(uv, vec4f(1.0, 0.0, 0.0, 1.0), vec2f(0.25, 0.75));
    let r1 = copy_at(uv, vec4f(0.0, -1.0, 1.0, 0.0), vec2f(0.5, 0.125));
    let r2 = copy_at(uv, vec4f(-1.0, 0.0, 0.0, -1.0), vec2f(0.625, 0.375));
    let r3 = copy_at(uv, vec4f(0.0, 1.0, -1.0, 0.0), vec2f(0.125, 0.625));
    let m = clamp(load_mask(c).r, 0.0, 1.0);
    let m90 = clamp(sample_mask(vec2f(uv.y, 1.0 - uv.x)).r, 0.0, 1.0);
    store(c, gray4(mix(mix(r2, r3, m), mix(r1, a, m), m90)));
}
