// The anti-aliased color with the input's alpha, or the anti-aliased alpha (Affect Alpha Channel).
fn effect(c: vec2i) -> vec4f {
    let a = select(load_input(c).a, load_alpha(c).r, p.affect_alpha_channel != 0);
    return vec4f(load_smooth(c).rgb, a);
}
