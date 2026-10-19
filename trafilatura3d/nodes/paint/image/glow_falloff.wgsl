// Precise glow: 1 inside the shape, a linear falloff to 0 at Width (pixels at 256) outside.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let g = clamp(1.0 - max(edge_sd(c), 0.0) / max(p.width * px256(), 1e-3), 0.0, 1.0);
    store_out(c, vec4f(g, g, g, 1.0));
}
