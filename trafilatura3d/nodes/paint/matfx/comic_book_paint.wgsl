// Flat comic paint: the warped, corrected base color shaded by AO and curvature, its value cut into 4 bands.
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let w = vec2f(mx_fbm(uv, 8, 3, 0.5, p.seed_ + 1.0), mx_fbm(uv, 8, 3, 0.5, p.seed_ + 2.0)) - 0.5;
    let wuv = uv + w * p.warp * 0.03;
    var col = mx_hsl_adjust(mx_contrast3(sample_basecolor(wuv).rgb, p.input_contrast), 0.0, 0.5, p.input_luminosity);
    if (p.input_type == 1) { col = vec3f(mx_luma(col)); }
    let shade = sample_baked_ambient_occlusion(wuv).r * clamp(1.0 + (sample_curvature(wuv).r - 0.5) * 1.5, 0.3, 1.5);
    var hsv = mx_rgb2hsv(col);
    let v = mx_sat(hsv.z * shade);
    let bands = 4.0;
    let x = v * bands;
    let f = fract(x);
    let q = (floor(x) + smoothstep(0.45, 0.55, f)) / bands;  // flat bands, a thin smooth step between them
    hsv.y = mx_sat(hsv.y * p.color_intensity);
    hsv.z = mx_sat(q + 0.5 / bands);
    let flat_col = mx_hsv2rgb(hsv);
    let a = load_basecolor(c).a;
    store_paint(c, vec4f(mix(col * shade, flat_col, p.paint_intensity), a));
}
