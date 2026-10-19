// Quantize Color: colours in the distance space (Lab / 100, or RGB).
fn to_space(rgb: vec3f) -> vec3f {
    let x = clamp(rgb, vec3f(0.0), vec3f(1.0));
    if (p.distance_color_space == 0) { return rgb_to_lab(x) / 100.0; }
    return x;
}
fn from_space(v: vec3f) -> vec3f {
    if (p.distance_color_space == 0) { return clamp(lab_to_rgb(v * 100.0), vec3f(0.0), vec3f(1.0)); }
    return v;
}
fn dist2(a: vec3f, b: vec3f) -> f32 { let e = a - b; return dot(e, e); }
