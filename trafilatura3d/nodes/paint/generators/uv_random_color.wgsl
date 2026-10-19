fn hsv(h: f32, s: f32, v: f32) -> vec3f {
    let k = clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
    return v * mix(vec3f(1.0), k, s);
}

fn effect(c: vec2i) -> vec4f {
    let t = load_uv_islands(c);
    if (t.a < 0.5) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let q = vec2i(round(t.rg * 1048575.0));  // the island's two random numbers, as integers
    let r = vec3f(hash21(q, p.seed_), hash21(q, p.seed_ + 17.0), hash21(q, p.seed_ + 29.0));
    var rgb = hsv(r.x, 0.45 + 0.5 * r.y, 0.55 + 0.45 * r.z);
    if (p.color_source_mode == 1 && CUSTOM_GRADIENT_CONNECTED) {
        let w = f32(size_custom_gradient().x);
        rgb = sample_custom_gradient(vec2f((0.5 + r.x * (w - 1.0)) / w, 0.5)).rgb;  // never wraps around
    }
    return vec4f(rgb, 1.0);
}
