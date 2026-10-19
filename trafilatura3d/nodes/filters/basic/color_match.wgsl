@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    let o = vec4f(load_color(c).rgb, v.a);
    let mi = load_mask(c).r;
    var m = clamp(load_selection(c).r, 0.0, 1.0);
    // Smoothness without Blur: the majority cut of the smoothed selection is done here.
    if (p.mask_smoothness > 0.0 && p.mask_blur <= 0.0) { m = majority_cut(c); }
    if (p.mask_mode == 1) { m = mi; }
    store_output(c, select(o, mix(v, o, m), p.use_mask != 0));
    store_mask(c, gray4(m));
}
