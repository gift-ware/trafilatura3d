@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    var g = dot(v, p.channel_weights);
    if (p.flatten_alpha != 0) { g = mix(p.background_value, g, v.a); }
    store(c, vec4f(g, g, g, 1.0));
}
