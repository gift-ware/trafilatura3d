fn effect(c: vec2i) -> vec4f {
    let reach = max(p.distance * mix(1.0, load_distance_map(c).r, p.distance_map_multiplier) * px256(), 1e-3);
    let sd = edge_sd(c) - p.curve_offset * reach;  // < 0 inside; Curve Offset > 0 moves the border outward
    var t = clamp(-sd / reach, 0.0, 1.0);                                   // Inward
    if (p.direction == 1) { t = clamp(1.0 - sd / reach, 0.0, 1.0); }        // Outward
    if (p.direction == 2) { t = clamp(0.5 - 0.5 * sd / reach, 0.0, 1.0); }  // Both
    let h = select(sqrt(max(1.0 - (1.0 - t) * (1.0 - t), 0.0)), 1.0 - sqrt(max(1.0 - t * t, 0.0)), p.curve_shape == 1);
    return vec4f(h, h, h, load_input(c).a);
}
