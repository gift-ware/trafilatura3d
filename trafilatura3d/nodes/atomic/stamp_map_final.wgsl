@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var bg = p.background;
    if (BACKGROUND_CONNECTED) { bg = load_background(c); }
    let l = load_layer(c);
    var o = vec4f(bg.rgb + l.rgb, clamp(bg.a + l.a, 0.0, 1.0));
    if (p.clamped != 0) { o = clamp(o, vec4f(0.0), vec4f(1.0)); }
    if (OUT_OUTPUT_GRAY) { o = vec4f(o.r, o.r, o.r, 1.0); }
    store(c, o);
}
