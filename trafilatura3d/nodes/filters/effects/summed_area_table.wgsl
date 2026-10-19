@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    if (p.pass_ == 0) { store(c, load_input(c) / (p.size_.x * p.size_.y)); return; }
    store(c, texel_input(c));
}
