@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let shape = shape_layer(c, p.mask_color, p.input_is_pre_multiplied == 1);
    // The blurred coverage, read from where the shadow comes from (gray: the mask, color: its alpha).
    let t = sample_blur(fx_uv(c) + p.offset);
    let cover = select(t.a, t.r, BLUR_GRAY);
    let s = clamp(cover / max(1.0 - p.spread, 1e-3), 0.0, 1.0) * p.opacity;
    let o = over(shape, vec4f(p.shadow_color.rgb, s * p.shadow_color.a));
    store_output(c, finish(o, p.pre_multiply_output == 1));
    store_mask(c, fx_gray(s * (1.0 - shape.a)));
}
