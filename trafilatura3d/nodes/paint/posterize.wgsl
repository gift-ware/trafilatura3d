fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let n = max(round(p.colors_amount), 2.0) - 1.0;
    let q = floor(v * n + 0.5) / n;
    return vec4f(q.rgb, select(v.a, q.a, p.apply_to_alpha != 0));
}
