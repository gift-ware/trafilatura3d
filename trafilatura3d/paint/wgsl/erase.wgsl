// Erase along path: the painting below loses the path's coverage (top alpha).

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    let b = load_below(c);
    store(c, vec4f(b.rgb, b.a * (1.0 - clamp(load_top(c).a, 0.0, 1.0))));
}
