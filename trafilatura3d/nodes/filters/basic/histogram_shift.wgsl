@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = p.position - floor(p.position);
    let v = load_input(c).r + s;
    store(c, gray4(select(v, v - 1.0, v > 1.0)));
}
