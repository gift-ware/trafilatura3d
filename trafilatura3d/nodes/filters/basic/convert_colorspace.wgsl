@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    let o = select(linear_to_srgb(v.rgb), srgb_to_linear(v.rgb), TO_LINEAR);
    store(c, vec4f(o, v.a));
}
