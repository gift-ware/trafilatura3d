@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // Contrast 0 leaves the sum as it is, 1 stretches it x5 around mid-grey.
    let v = fractal_sum_at(c, p.non_square_expansion, p.min_level, p.max_level, p.roughness,
                           1.0 + 4.0 * clamp(p.contrast, 0.0, 1.0), p.global_opacity, p.seed_,
                           p.disorder * p.disorder_speed);
    store(c, vec4f(v, v, v, 1.0));
}
