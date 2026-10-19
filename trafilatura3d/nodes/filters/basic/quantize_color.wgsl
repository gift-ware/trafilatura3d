fn bayer8(q: vec2i) -> f32 {
    let x = u32(q.x) & 7u;
    let y = u32(q.y) & 7u;
    let v = x ^ y;
    let b = ((v & 1u) << 5u) | ((y & 1u) << 4u) | ((v & 2u) << 2u) | ((y & 2u) << 1u) | ((v & 4u) >> 1u) | ((y & 4u) >> 2u);
    return (f32(b) + 0.5) / 64.0;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    let s = to_space(v.rgb);
    var b1 = 0;
    var b2 = 0;
    var d1 = 1e30;
    var d2 = 1e30;
    for (var k = 0; k < p.count; k++) {
        let e = texel_palette(vec2i(k, 0));
        if (e.a <= 0.0 && k > 0) { continue; }
        let dd = dist2(s, e.rgb);
        if (dd < d1) { d2 = d1; b2 = b1; d1 = dd; b1 = k; }
        else if (dd < d2) { d2 = dd; b2 = k; }
    }
    var pick = b1;
    if (p.dithering > 0.0 && d2 < 1e29) {
        // Position between the two nearest colours (0 at the nearest, 0.5 halfway).
        let a = texel_palette(vec2i(b1, 0)).rgb;
        let b = texel_palette(vec2i(b2, 0)).rgb;
        let ab = b - a;
        let f = clamp(dot(s - a, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
        var t = fract(f32(c.x) * 0.7548776662 + f32(c.y) * 0.5698402910);
        if (p.dithering_pattern == 1) { t = bayer8(c); }
        if (f > mix(0.5, t, clamp(p.dithering, 0.0, 1.0))) { pick = b2; }
    }
    let n = p.count;
    store_output(c, vec4f(from_space(texel_palette(vec2i(pick, 0)).rgb), v.a));
    store_id(c, gray4(f32(pick) / f32(max(n - 1, 1))));
    let slot = min(i32(floor(uv_at(c).x * f32(n))), n - 1);
    let e = texel_palette(vec2i(slot, 0));
    store_palette(c, select(vec4f(0.0), vec4f(from_space(e.rgb), 1.0), e.a > 0.0));
    store_palette_color_amount(c, vec4f(f32(n), 0.0, 0.0, 0.0));
}
