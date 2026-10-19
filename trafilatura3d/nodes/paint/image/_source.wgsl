// Source of Warp and Blur Slope: s_mode 0 the default noise (tileable fbm, 8 cells per repetition), 1 the input's
// luminance, 2 the Custom Noise input; repeated s_tiling times (whole repetitions, so the result tiles), the
// midpoint shifted by s_balance (0.5 = unchanged) and contrasted by s_contrast.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let t = max(round(p.s_tiling), 1.0);
    let uv = fract(uv_of(c) * t);
    var v = pfbm(uv_of(c), 8 * i32(t), p.seed_);
    if (p.s_mode == 1) {
        let s = sample_input(uv);
        v = select(luminance(s.rgb), s.r, INPUT_GRAY);
    } else if (p.s_mode == 2) {
        v = sample_custom_noise(uv).r;
    }
    v = contrasted(clamp(v + p.s_balance - 0.5, 0.0, 1.0), p.s_contrast);
    store_out(c, vec4f(v, v, v, 1.0));
}
