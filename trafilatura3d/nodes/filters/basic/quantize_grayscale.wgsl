@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store(c, gray4(quantize_steps(load_input(c), vec4f(f32(p.quantize))).r));
}
