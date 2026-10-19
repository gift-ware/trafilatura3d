@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_input(c);
    let flip = vec4f(f32(p.invert_red), f32(p.invert_green), f32(p.invert_blue), f32(p.invert_alpha));
    store(c, mix(t, 1.0 - t, flip));
}
