// The color of the nearest stroke pixel, averaged over Blur Samples points of a golden-angle spiral filling a disc
// of Blur Intensity around the pixel.
fn blurred(c: vec2i) -> vec4f {
    let n = max(p.blur_samples, 1);
    let r = p.blur_intensity * px256();
    var acc = vec3f(0.0);
    var w = 0.0;
    for (var k = 0; k < n; k++) {
        let a = f32(k) * 2.39996323;
        let q = clamp(c + vec2i(round(r * sqrt((f32(k) + 0.5) / f32(n)) * vec2f(cos(a), sin(a)))), vec2i(0), size() - 1);
        let s = texel_seeds(q);
        if (s.x < 0.0) { continue; }
        acc += load_input(vec2i(s.xy)).rgb;
        w += 1.0;
    }
    return vec4f(acc / max(w, 1.0), w);
}

fn effect(c: vec2i) -> vec4f {
    let t = load_input(c);
    let o = texel_outline(c).r > 0.5;
    var inside = !o && texel_scan(c).r < 0.5;
    if (p.uv_border_behavior == 1) { inside = inside && load_uv_islands(c).a >= p.uv_border_threshold; }
    if (p.debug_area_detection != 0) {
        let g = select(select(0.0, 0.35, inside), 1.0, o);
        return vec4f(g, g, g, 1.0);
    }
    if (!inside) { return t; }
    var col = p.fill_color;
    if (p.color_mode == 1) {
        let b = blurred(c);
        if (b.a > 0.0) { col = b.rgb; }
    } else if (p.color_mode == 2) {
        let d = texel_colors(c);
        if (d.a > 0.0) { col = d.rgb; }
    }
    return vec4f(col, 1.0);
}
