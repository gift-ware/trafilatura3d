@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let o = sample_$OPACITY((vec2f(c) + 0.5) / vec2f(d)).r;
    store(c, select(vec4f(-1.0, -1.0, 0.0, 1.0), vec4f(f32(c.x), f32(c.y), 0.0, 1.0), o > 0.5));
}
