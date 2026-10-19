fn effect(c: vec2i) -> vec4f {
    let n = normalize(load_world_space_normals(c).rgb * 2.0 - 1.0);
    let h = radians(p.horizontal_angle); let v = radians(p.vertical_angle);
    let l = normalize(vec3f(cos(v) * sin(h), sin(v), cos(v) * cos(h)));
    let lambert = max(dot(n, l), 0.0);
    let spec = pow(lambert, mix(2.0, 64.0, p.highlight_glossiness)) * p.highlight_level * 2.0;
    var x = clamp(lambert * (1.0 - p.highlight_level * 0.5) + spec, 0.0, 1.0);
    if (POSITION_CONNECTED) {  // attenuation along the light: farther from the light, darker
        let d = dot(load_position(c).rgb - 0.5, l) + 0.5;
        x *= mix(1.0, clamp(d, 0.0, 1.0), p.light_attenuation);
    }
    if (p.invert != 0) { x = 1.0 - x; }
    return vec4f(vec3f(x), 1.0);
}
