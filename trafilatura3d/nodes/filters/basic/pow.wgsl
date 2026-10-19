@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    store(c, vec4f(sign(v.rgb) * pow(abs(v.rgb), vec3f(max(p.exponent, 0.0))), v.a));
}
