@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let h = cube_hit(c);
    if (!h.hit) {
        store_depth(c, vec4f(0.0, 0.0, 0.0, 1.0));
        store_position(c, vec4f(1.0, 1.0, 1.0, 1.0));
        store_normal(c, vec4f(0.5, 0.5, 1.0, 1.0));
        return;
    }
    store_depth(c, vec4f(h.depth, h.depth, h.depth, 1.0));
    store_position(c, vec4f(h.obj.zxy, 1.0));                  // the object frame is z-up
    store_normal(c, vec4f(h.normal * 0.5 + 0.5, 1.0));
}
