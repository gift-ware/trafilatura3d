fn effect(c: vec2i) -> vec4f {
    let t = load_input(c);
    let m = select(luminance(t.rgb), t.r, INPUT_GRAY);
    let s = spread(load_glow(c).r, p.spread) * p.opacity;
    return behind(t, m, s, p.color, p.mask_ != 0);
}
