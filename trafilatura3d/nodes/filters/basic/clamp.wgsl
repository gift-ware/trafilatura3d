@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    let o = min(max(v, vec4f(p.min)), vec4f(p.max));
    store(c, vec4f(o.rgb, select(v.a, o.a, p.apply_to_alpha != 0)));
}
