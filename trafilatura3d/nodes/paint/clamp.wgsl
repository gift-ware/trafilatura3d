fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    return vec4f(clamp(v.rgb, vec3f(p.min), vec3f(p.max)), v.a);
}
