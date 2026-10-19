@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let g = frame(c, p.non_square_expansion) * f32(max(p.tiling, 1));
    // Four ropes per tile, eight tilted strands each (sizes and angle tuned).
    let v = strands(g, vec2f(0.125, 0.25), vec2f(0.0625, 0.0), vec2f(0.1858, 0.1256), radians(34.3), false);
    store(c, vec4f(v, v, v, 1.0));
}
