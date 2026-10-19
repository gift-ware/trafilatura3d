@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let e = pow(convex(load_curvature(c).r), mix(2.5, 0.4, p.hard_edges_amount));
    store(c, g4(e));
}
