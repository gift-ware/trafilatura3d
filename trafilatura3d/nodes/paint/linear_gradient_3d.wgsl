fn effect(c: vec2i) -> vec4f {
    let pos = load_position(c).rgb;
    let ab = p.position_end - p.position_start;
    var t = clamp(dot(pos - p.position_start, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    t = pow(t, log(0.5) / log(clamp(p.balance, 1e-3, 1.0 - 1e-3)));
    t = contrasted(t, p.contrast);
    if (p.invert != 0) { t = 1.0 - t; }
    return vec4f(vec3f(t), 1.0);
}
