// Worn areas (blurred generator mask, leveled by Generator Mask Control / Contrast) take the Material group's
// values; the paint's step around them goes into the normal (Normal Intensity).
fn wear(uv: vec2f) -> f32 {
    return mx_scan(sample_mask(uv).r, p.generator_mask_control, p.generator_mask_contrast);
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.0 / p.size_;
    let w = wear(uv);
    let bc = load_basecolor(c);
    store_basecolor(c, vec4f(mix(bc.rgb, p.base_color.rgb, w), bc.a));
    let m = load_metallic(c);
    store_metallic(c, vec4f(vec3f(mix(m.r, p.metallic, w)), m.a));
    let r = load_roughness(c);
    store_roughness(c, vec4f(vec3f(mix(r.r, p.roughness, w)), r.a));
    let d = load_diffuse(c);
    store_diffuse(c, vec4f(mix(d.rgb, p.diffuse.rgb, w), d.a));
    let s = load_specular(c);
    store_specular(c, vec4f(mix(s.rgb, p.specular_color.rgb, w), s.a));
    let g = load_glossiness(c);
    store_glossiness(c, vec4f(vec3f(mix(g.r, p.glossiness, w)), g.a));
    let k = p.normal_intensity * 3.0 * mx_px();
    let slope = mx_nslope(-(wear(uv + vec2f(o.x, 0.0)) - w), -(wear(uv - vec2f(0.0, o.y)) - w), k);
    let n = load_normal(c);
    store_normal(c, vec4f(mx_nenc(mx_ndetail(mx_ndec(n.rgb), slope)), n.a));
}
