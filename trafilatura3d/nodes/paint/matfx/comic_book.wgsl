// Ink borders on the paint's edges and the cavities, sharpening, output correction.
fn lum_at(uv: vec2f) -> f32 { return mx_luma(sample_paint(uv).rgb); }

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.5 * mx_px() / p.size_;
    let col = sample_paint(uv);
    let l = lum_at(uv + vec2f(-o.x, 0.0)); let r = lum_at(uv + vec2f(o.x, 0.0));
    let u = lum_at(uv + vec2f(0.0, -o.y)); let d = lum_at(uv + vec2f(0.0, o.y));
    let grad = length(vec2f(r - l, d - u));
    let cavity = mx_sat((0.5 - load_curvature(c).r) * 4.0) * (1.0 - load_baked_ambient_occlusion(c).r * 0.5);
    let e = max(grad * 2.0, cavity);
    let t = (1.0 - p.border_tolerance) * 0.4;
    let ink = smoothstep(t, t + 0.02 + p.border_smoothness * 0.4, e) * p.border_opacity;
    let sharp = col.rgb + (col.rgb - (sample_paint(uv + vec2f(-o.x, 0.0)).rgb + sample_paint(uv + vec2f(o.x, 0.0)).rgb
                                    + sample_paint(uv + vec2f(0.0, -o.y)).rgb + sample_paint(uv + vec2f(0.0, o.y)).rgb) * 0.25) * p.sharpness * 2.0;
    var res = mix(sharp, vec3f(0.02), ink);
    res = mx_hsl_adjust(mx_contrast3(res, p.output_contrast), 0.0, 0.5, p.output_luminosity);
    store_basecolor(c, vec4f(res, col.a));
}
