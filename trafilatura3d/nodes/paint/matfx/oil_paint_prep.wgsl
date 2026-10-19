// The corrected source and its structure tensor (gx², gy², gx·gy of the luminance, gradient length).
fn source(uv: vec2f) -> vec4f {
    var s = sample_basecolor(uv);
    if (p.input_type == 1) { s = sample_image_input(uv); }
    let rgb = mx_hsl_adjust(mx_contrast3(s.rgb, p.input_image_contrast), p.input_image_hue,
                            p.input_image_saturation, p.input_image_luminosity);
    return vec4f(rgb, s.a);
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = mx_px() / p.size_;
    let gx = mx_luma(source(uv + vec2f(o.x, 0.0)).rgb) - mx_luma(source(uv - vec2f(o.x, 0.0)).rgb);
    let gy = mx_luma(source(uv + vec2f(0.0, o.y)).rgb) - mx_luma(source(uv - vec2f(0.0, o.y)).rgb);
    store_src(c, source(uv));
    store_tensor(c, vec4f(gx * gx, gy * gy, gx * gy, sqrt(gx * gx + gy * gy)));
}
