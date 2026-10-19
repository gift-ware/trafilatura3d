fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    return select(v, vec4f(1.0 - v.rgb, v.a), p.invert != 0);
}
