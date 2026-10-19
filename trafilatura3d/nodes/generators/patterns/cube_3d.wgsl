@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let h = cube_hit(c);
    let v = select(0.0, h.depth, h.hit);
    store(c, vec4f(v, v, v, 1.0));
}
