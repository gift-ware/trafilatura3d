fn effect(c: vec2i) -> vec4f {
    let t = clamp(load_texel_density(c).r, -1.0, 1.0);  // octaves from the mean
    let rgb = select(mix(p.color_medium, p.color_high, t), mix(p.color_medium, p.color_low, -t), t < 0.0);
    return vec4f(rgb, 1.0);
}
