fn effect(c: vec2i) -> vec4f {
    let s = load_state(c);
    let g = pow(clamp(select(luminance(s.rgb), s.r, STATE_GRAY), 0.0, 1.0), 1.0 + 7.0 * p.contrast);
    return vec4f(g, g, g, load_input(c).a);
}
