fn alpha_levels(b: f32) -> f32 {
    let h = clamp(p.hardness, 0.0, 1.0);
    if (h <= 0.0) { return select(0.0, 1.0, b >= 1.0); }  // no ramp left: a step, the input white included
    if (h <= 0.5) { return clamp((b - (1.0 - 2.0 * h)) / (2.0 * h), 0.0, 1.0); }
    return clamp(b / max(2.0 * (1.0 - h), 1e-6), 0.0, 1.0);
}

fn effect(c: vec2i) -> vec4f {
    var b = load_input(c).r;
    if (p.invert != 0) { b = 1.0 - b; }
    let t = 1.0 - abs(uv_of(c) * 2.0 - 1.0);
    let s = max(p.shape, 1e-6);
    let m = select(1.0, smoothstep(0.0, 1.0, t.x / s) * smoothstep(0.0, 1.0, t.y / s), p.shape > 0.0);
    var v: f32;
    if (p.order == 0) { v = clamp(alpha_levels(b) + m - 1.0, 0.0, 1.0); }
    else { v = alpha_levels(clamp(b + m - 1.0, 0.0, 1.0)); }
    if (p.threshold != 0) { v = select(0.0, 1.0, v > 0.5 / 255.0); }
    return vec4f(v, v, v, 1.0);
}
