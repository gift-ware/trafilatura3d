// Quantize, final pass: each pixel (dithered) takes the nearest color of the palette (clusters with pixels).

// Ordered dither threshold in [0, 1) of a Bayer matrix of 2^levels x 2^levels.
fn bayer(q: vec2i, levels: i32) -> f32 {
    var v = 0;
    for (var i = 0; i < levels; i++) {
        let x = (q.x >> u32(i)) & 1;
        let y = (q.y >> u32(i)) & 1;
        v += (((x ^ y) << 1u) | y) << u32(2 * (levels - 1 - i));
    }
    return (f32(v) + 0.5) / f32(1 << u32(2 * levels));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let k_max = clamp(p.color_amount, 2, 64);
    var used = 0;
    for (var k = 0; k < k_max; k++) { if (texel_palette(vec2i(k, 0)).w > 0.0) { used++; } }
    let v = load_input(c);
    // Dither: an offset of the color before the palette lookup, as large as the palette's spacing.
    let q = vec2i(floor(vec2f(c) * 256.0 / p.size_.x));  // pattern cells: pixels at 256 px
    var t = hash21(q, p.seed_);
    if (p.dithering_pattern < 3) { t = bayer(q, p.dithering_pattern + 1); }
    let spread = 1.0 / pow(f32(max(used, 1)), 1.0 / 3.0);
    let x = to_space(clamp(v.rgb + (t - 0.5) * p.dithering * spread, vec3f(0.0), vec3f(1.0)), p.distance_color_space);
    var best = vec3f(x);
    var dist = 1e30;
    for (var k = 0; k < k_max; k++) {
        let e = texel_palette(vec2i(k, 0));
        let d = dot(x - e.xyz, x - e.xyz);
        if (e.w > 0.0 && d < dist) { dist = d; best = e.xyz; }
    }
    let a = select(v.a, select(0.0, 1.0, v.a >= p.alpha_threshold), p.apply_to_alpha != 0);
    store_output(c, vec4f(from_space(best, p.distance_color_space), a));
    store_palette_color_amount(c, vec4f(f32(used), f32(used), f32(used), 1.0));
}
