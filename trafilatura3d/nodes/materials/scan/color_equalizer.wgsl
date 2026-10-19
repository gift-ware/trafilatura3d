@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let v = load_input(c);
    let m = load_m(c).r;
    store_output(c, vec4f(equalize(v.rgb, sample_blurred(uv).rgb, load_avg(c).rgb, m), v.a));
    store_mask(c, vec4f(m, m, m, 1.0));
}
