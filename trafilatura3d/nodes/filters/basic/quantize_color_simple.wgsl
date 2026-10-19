@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var n = vec4f(f32(p.quantize_red_channel), f32(p.quantize_green_channel), f32(p.quantize_blue_channel),
                  f32(p.quantize_alpha_channel));
    if (INPUT_GRAY) { n = vec4f(n.x); }
    var o = quantize_steps(load_input(c), n);
    if (INPUT_GRAY) { o = gray4(o.r); }
    store(c, o);
}
