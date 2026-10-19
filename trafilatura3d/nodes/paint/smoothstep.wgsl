fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    return vec4f(smoothstep(vec3f(p.minimum), vec3f(max(p.maximum, p.minimum + 1e-4)), v.rgb), v.a);
}
