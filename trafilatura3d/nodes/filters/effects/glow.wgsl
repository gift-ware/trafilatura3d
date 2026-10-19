@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_input(c);
    let g = clamp(texel_blur(c).r, 0.0, 1.0) * p.glow_amount;
    if (OUT_OUTPUT_GRAY) { store(c, fx_gray(t.r + g)); return; }
    store(c, vec4f(t.rgb + p.glow_color.rgb * p.glow_color.a * g, t.a));
}
