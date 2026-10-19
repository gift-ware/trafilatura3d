fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    return vec4f(fract(v.rgb + p.position), v.a);
}
