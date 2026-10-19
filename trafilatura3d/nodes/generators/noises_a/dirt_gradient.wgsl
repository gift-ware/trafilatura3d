@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let uv = uv_of(c);
    let per0 = vec2i(round(4.0 * asp));
    // Grainy clouds down to the pixel (octaves counted on the short side, as in rough.wgsl).
    let oct = auto_octaves(4.0 * max(asp.x, asp.y), 12) + 1;
    let z = fbm_z(uv, per0, oct, 0.85, p.seed_, 0, p.disorder * p.disorder_speed, p.disorder_anisotropy,
                  p.disorder_anisotropy_angle);
    // Falloff: almost black at the top, dirt rising towards the bottom; the grain modulates it.
    let fall = 0.053 + 0.603 * smoothstep(0.27, 1.05, uv.y);
    let v = clamp(fall * (1.0 + 0.105 * z) + 0.032 * z, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
