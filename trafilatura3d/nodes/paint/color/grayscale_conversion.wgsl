fn effect(c: vec2i) -> vec4f {
    let layer = load_input(c);
    let custom = load_custom_input(c);
    let v = clamp(select(layer, custom, p.channel_input == 1), vec4f(0.0), vec4f(1.0));
    var g = 0.0;
    if (p.mode == 5) {
        g = clamp(dot(v, vec4f(p.red, p.green, p.blue, p.alpha)), 0.0, 1.0);
    } else {
        g = gray_of(v.rgb, p.mode);
    }
    return vec4f(vec3f(balance_contrast(g, p.balance, p.contrast, p.invert)), 1.0);
}
