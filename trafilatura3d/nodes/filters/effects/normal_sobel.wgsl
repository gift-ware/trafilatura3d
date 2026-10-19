@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = fx_input_slopes(c, true) * p.intensity / 25.6;
    store(c, vec4f(fx_encode_normal(fx_slope_normal(s), p.normal_format == 1), 1.0));
}
