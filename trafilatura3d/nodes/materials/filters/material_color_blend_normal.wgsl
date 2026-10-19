fn src_h(uv: vec2f) -> f32 {
    if (p.source == 1) { return sample_greyscale_mask(uv).r; }
    return sample_height(uv).r;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = 1.0 / vec2f(d);
    var m = load_greyscale_mask(c).r;
    if (p.color_id_mask != 0) { m = id_mask(load_colorid(c).rgb, p.color.rgb, p.fuzziness, p.padding); }
    let dx = (src_h(uv + vec2f(e.x, 0.0)) - src_h(uv - vec2f(e.x, 0.0))) * 0.5;
    let dy = (src_h(uv - vec2f(0.0, e.y)) - src_h(uv + vec2f(0.0, e.y))) * 0.5;  // rows grow downwards
    let k = p.normal_height_intensity * 40.0 * f32(d.x) / 256.0;
    let gen = n_slopes(dx, dy, k);
    let dxf = p.format == 0;
    let base = n_dec(load_normal(c).rgb, dxf);
    var n = n_combine(base, gen);
    if (p.normal_blending_mode == 1) { n = gen; }
    // With the Mask as source the relief is the mask's own; with Height it is applied through the mask.
    var a = p.normal_height_opacity;
    if (p.source == 0) { a *= sat(m); }
    store_normal(c, out_normal(c, vec4f(n_enc(n_lerp(base, n, a), dxf), 1.0)));
}
