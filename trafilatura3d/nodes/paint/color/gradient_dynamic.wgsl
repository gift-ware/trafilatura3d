fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let t = clamp(select(luminance(v.rgb), v.r, INPUT_GRAY), 0.0, 1.0);
    // Sample at texel centers along the line, so 0 and 1 are the source's first and last texels.
    let n = vec2f(size_gradient_source());
    let along = (0.5 + t * (select(n.x, n.y, p.gradient_orientation == 1) - 1.0)) / select(n.x, n.y, p.gradient_orientation == 1);
    let uv = select(vec2f(along, p.gradient_input_position), vec2f(p.gradient_input_position, along),
                    p.gradient_orientation == 1);
    let g = sample_gradient_source(uv);
    if (!GRADIENT_SOURCE_CONNECTED) { return vec4f(vec3f(t), v.a); }
    return vec4f(g.rgb, v.a);
}
