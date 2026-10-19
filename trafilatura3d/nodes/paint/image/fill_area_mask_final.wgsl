fn effect(c: vec2i) -> vec4f {
    let t = load_input(c);
    let o = texel_outline(c).r > 0.5;
    var inside = !o && texel_scan(c).r < 0.5;
    if (p.uv_border_behavior == 1) { inside = inside && load_uv_islands(c).a >= p.uv_border_detection_threshold; }
    if (p.debug_area_detection != 0) {
        let g = select(select(0.0, 0.35, inside), 1.0, o);
        return vec4f(g, g, g, 1.0);
    }
    if (inside) { return vec4f(1.0); }
    return t;
}
