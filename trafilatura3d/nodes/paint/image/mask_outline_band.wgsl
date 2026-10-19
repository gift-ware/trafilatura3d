fn effect(c: vec2i) -> vec4f {
    let sd = edge_sd(c);
    var d = max(sd, 0.0);                        // Outside
    if (p.mask_position == 1) { d = max(-sd, 0.0); }  // Inside
    if (p.mask_position == 2) { d = abs(sd); }        // Both
    let t = clamp(d / max(p.width * px256(), 1e-3), 0.0, 1.0);
    let on_side = select(sd >= 0.0, sd <= 0.0, p.mask_position == 1) || p.mask_position == 2;
    var v = select(0.0, 1.0 - pow(t, exp2((1.0 - 2.0 * p.curve_shape) * 3.0)), on_side && t < 1.0);
    if (p.start_from_black != 0) { v = 1.0 - v; }
    return vec4f(v, v, v, load_input(c).a);
}
