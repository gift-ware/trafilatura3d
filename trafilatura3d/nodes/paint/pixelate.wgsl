fn effect(c: vec2i) -> vec4f {
    let n = max(round(p.pixels_amount), 1.0);
    let q = sample_input((floor(uv_of(c) * n) + 0.5) / n);
    let v = load_input(c);
    return vec4f(q.rgb, select(v.a, q.a, p.apply_to_alpha != 0));
}
