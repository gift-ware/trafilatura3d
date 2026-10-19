@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    store(c, vec4f(select(v.rgb, v.rgb / v.a, v.a > 1e-6), 1.0));
}
