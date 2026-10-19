// Blur HQ's Quality 1: the two chains of directional blurs averaged.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    store(c, 0.5 * (load_a(c) + load_b(c)));
}
