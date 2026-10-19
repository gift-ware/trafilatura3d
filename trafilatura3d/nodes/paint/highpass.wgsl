fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let h = v - load_blurred(c) + 0.5;
    return vec4f(h.rgb, select(v.a, h.a, p.apply_to_alpha != 0));
}
