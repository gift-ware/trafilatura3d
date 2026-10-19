// The Splatter's layer: final_ 0 writes the Background Color into the layer the tiles are drawn over
// (splatter_draw.wgsl); 1 writes the layer through Output Level Min/Max (input levels) as the output.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    if (p.final_ == 0) {
        var bg = vec4f(p.background_color.rgb, 1.0);
        if (sp_gray()) { bg = vec4f(vec3f(luminance(bg.rgb)), 1.0); }
        store(c, bg);
        return;
    }
    let o = load_layer(c);
    let span = p.output_level_max - p.output_level_min;
    let r = clamp((o.rgb - p.output_level_min) / select(span, 1e-6, abs(span) < 1e-6), vec3f(0.0), vec3f(1.0));
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(r.r, r.r, r.r, 1.0)); } else { store(c, vec4f(r, 1.0)); }
}
