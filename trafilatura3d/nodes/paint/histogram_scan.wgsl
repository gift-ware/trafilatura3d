fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let pos = select(p.position, 1.0 - p.position, p.invert_position != 0);
    let w = (1.0 - p.contrast) * 0.5;
    let lo = mix(-w, 1.0 - w, 1.0 - pos);
    let hi = lo + max(2.0 * w, 1e-4);
    return vec4f(clamp((v.rgb - lo) / (hi - lo), vec3f(0.0), vec3f(1.0)), v.a);
}
