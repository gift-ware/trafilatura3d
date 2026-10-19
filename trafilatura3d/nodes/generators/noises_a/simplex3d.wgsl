@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var u = unit_slice(c);
    if (POSITION_MAP_CONNECTED) { u = load_position_map(c).rgb; }
    let f = freq3(p.scale, p.size, true);
    let v = 0.5 + 0.65 * simplex3(u * f, per3(f, true), p.seed_);
    store(c, vec4f(v, v, v, 1.0));
}
