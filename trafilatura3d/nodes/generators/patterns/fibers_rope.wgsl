@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let g = frame(c, p.non_square_expansion) * f32(max(p.tiling, 1));
    // One rope per tile: four tilted Gaussian strands (sizes and angle tuned).
    let v = strands(g, vec2f(0.25, 1.0), vec2f(0.125, 0.5), vec2f(0.309, 0.2088), radians(33.1), true);
    store(c, vec4f(v, v, v, 1.0));
}
