fn effect(c: vec2i) -> vec4f {
    let t = load_input(c);
    let s = spread(load_shadow(c).r, p.spread) * p.opacity;
    return behind(t, t.r, s, p.color, INPUT_GRAY);
}
