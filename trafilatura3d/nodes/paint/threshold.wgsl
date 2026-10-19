fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let x = luminance(v.rgb);
    var on: bool;
    switch p.mode {
        case 1: { on = x >= p.threshold; }
        case 2: { on = x < p.threshold; }
        case 3: { on = x <= p.threshold; }
        default: { on = x > p.threshold; }
    }
    return vec4f(vec3f(select(0.0, 1.0, on)), v.a);
}
