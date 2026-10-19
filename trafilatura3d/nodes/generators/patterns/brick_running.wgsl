@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let v = brick_simple(uv, f32(max(p.scale, 1)) / 4.0, 2.0, 4.0, p.interstice_width * 0.5, p.edge_smoothness * 0.5, false);
    store(c, vec4f(v, v, v, 1.0));
}
