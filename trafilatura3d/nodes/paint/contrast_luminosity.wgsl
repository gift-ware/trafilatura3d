fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let k = select(1.0 + p.contrast, 1.0 / max(1.0 - p.contrast, 1e-3), p.contrast > 0.0);
    return vec4f((v.rgb - 0.5) * k + 0.5 + p.luminosity, v.a);
}
