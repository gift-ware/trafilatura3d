fn effect(c: vec2i) -> vec4f {
    let d = distance(load_position(c).rgb, vec3f(p.position_x, p.position_y, p.position_z));
    var t = clamp((d - p.offset) / max(p.radius, 1e-4), 0.0, 1.0);
    t = contrasted(t, p.contrast);
    if (p.invert != 0) { t = 1.0 - t; }
    return vec4f(vec3f(t), 1.0);
}
