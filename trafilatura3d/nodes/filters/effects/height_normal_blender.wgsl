@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dx = p.normal_format == 0;
    let t = load_normal(c);
    let base = normalize(fx_decode_normal(t, dx));
    let detail = fx_slope_normal(fx_input_slopes(c, true) * p.normal_intensity / 25.6);
    store(c, vec4f(fx_encode_normal(fx_rnm(base, detail), dx), t.a));
}
