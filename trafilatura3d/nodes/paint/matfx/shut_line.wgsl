// Groove = the steps blurred over Cavity Width (renormalized so a thin step reaches 1), lips = the wider blur
// around it. Their relief goes into height and normal, the groove takes Line Color.
fn groove(uv: vec2f) -> f32 {
    let g = mx_sat(sample_groove(uv).r * max(p.cavity_width, 0.05) * 1.25);
    return smoothstep(0.0, 1.0 - 0.9 * p.slope_intensity, g);
}
fn relief(uv: vec2f) -> f32 {
    let g = groove(uv);
    let l = mx_sat(sample_lips(uv).r * max(p.cavity_width, 0.05) * 2.5 * 1.25);
    return (-g + max(l - g, 0.0) * p.emboss_strength) * p.height_intensity;
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.0 / p.size_;
    let g = groove(uv);
    let r = relief(uv);
    let bc = load_basecolor(c);
    let keep = max(p.source_material_opacity, max(g, mx_sat(abs(r) * 4.0)));
    store_basecolor(c, vec4f(mix(bc.rgb, p.line_color.rgb, g), bc.a * keep));
    let h = load_height(c);
    store_height(c, vec4f(vec3f(h.r + r * 0.1), h.a * keep));
    let k = 2.0 * mx_px();
    let slope = mx_nslope(relief(uv + vec2f(o.x, 0.0)) - r, relief(uv - vec2f(0.0, o.y)) - r, k);
    let n = load_normal(c);
    store_normal(c, vec4f(mx_nenc(mx_ndetail(mx_ndec(n.rgb), slope)), n.a * keep));
}
