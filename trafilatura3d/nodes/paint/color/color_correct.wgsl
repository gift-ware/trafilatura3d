// One tonal range's correction: contrast around `pivot` (positive steeper, negative flatter), luminosity added,
// saturation away from the luminance.
fn corrected(v: vec3f, pivot: f32, contrast: f32, lum: f32, sat: f32) -> vec3f {
    let k = select(1.0 + contrast, 1.0 / max(1.0 - contrast, 1e-3), contrast > 0.0);
    let x = (v - pivot) * k + pivot + lum * 0.5;
    let g = luminance(x);
    return g + (x - g) * (1.0 + sat);
}

fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let w = tone_weights(luminance(v.rgb));
    let o = w.x * corrected(v.rgb, 0.2, p.shadows_contrast, p.shadows_luminosity, p.shadows_saturation)
          + w.y * corrected(v.rgb, 0.5, p.midtones_contrast, p.midtones_luminosity, p.midtones_saturation)
          + w.z * corrected(v.rgb, 0.8, p.highlights_contrast, p.highlights_luminosity, p.highlights_saturation);
    return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), v.a);
}
