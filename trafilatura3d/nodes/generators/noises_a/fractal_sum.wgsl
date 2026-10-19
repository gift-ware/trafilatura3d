@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dis = p.disorder * p.disorder_speed;
    let nse = p.non_square_expansion;
    let s = p.seed_ + f32(VARIANT) * 57.0;
    // The presets: Fractal sum base with fixed levels, roughness, contrast factor and opacity.
    var v = 0.0;
    if (VARIANT == 0) { v = fractal_sum_at(c, nse, 0, 13, 1.1, 4.3, 0.77, s, dis); }
    else if (VARIANT == 1) { v = fractal_sum_at(c, nse, 0, 13, 1.16, 6.5, 0.62, s, dis); }
    else if (VARIANT == 2) { v = fractal_sum_at(c, nse, 6, 13, 0.54, 1.4, 1.0, s, dis); }
    else { v = fractal_sum_at(c, nse, 5, 13, 0.4, 1.0, 1.18, s, dis); }
    store(c, vec4f(v, v, v, 1.0));
}
