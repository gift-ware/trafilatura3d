@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = fx_input_slopes(c, p.sampling == 1) * p.height_depth_cm / max(p.surface_size_cm, 1e-4);
    store(c, vec4f(fx_encode_normal(fx_slope_normal(s), p.normal_format == 1), 1.0));
}
