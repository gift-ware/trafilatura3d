@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let v = hash21(c, p.seed_);
    store(c, vec4f(v, v, v, 1.0));
}
