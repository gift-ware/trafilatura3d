fn pattern(k: i32, q: vec2f) -> vec4f {
    switch k {
        case 0: { return sample_pattern_input_1(q); }
        case 1: { return sample_pattern_input_2(q); }
        case 2: { return sample_pattern_input_3(q); }
        case 3: { return sample_pattern_input_4(q); }
        case 4: { return sample_pattern_input_5(q); }
        case 5: { return sample_pattern_input_6(q); }
        case 6: { return sample_pattern_input_7(q); }
        default: { return sample_pattern_input_8(q); }
    }
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var bg = p.background_color;
    if (OUT_OUTPUT_GRAY) { bg = fx_gray(bg.r); }
    let n = size_flood_fill_bbox();
    let q = ff_pixel(c, n);
    let cell = ff_cell(texel_flood_fill_bbox(q), q, n);
    if (!cell.ok) { store(c, bg); return; }
    let stats = textureLoad(src_stats, vec2i(0), 0);
    let largest = max(stats.x, 1e-6);
    let smallest = max(-stats.y, 1e-6);
    let side = max(cell.size.x, cell.size.y);

    // Which pattern.
    let count = clamp(p.pattern_input_number, 1, 8);
    var t = ff_random(cell, 11.0);
    if (p.pattern_distribution_mode == 1) { t = side / largest; }
    if (p.pattern_distribution_mode == 2) { t = sample_pattern_distribution_map(cell.centre).r; }
    if (p.pattern_distribution_jittering > 0.0) { t = fract(t + p.pattern_distribution_jittering * ff_random(cell, 12.0)); }
    let k = clamp(i32(floor(t * f32(count))), 0, count - 1);

    // Extent of the pattern (UV).
    var ext = vec2f(1.0);
    if (p.size_mode == 1) { ext = vec2f(length(cell.size)); }
    if (p.size_mode == 2) { ext = vec2f(largest); }
    if (p.size_mode == 3) { ext = vec2f(smallest); }
    if (p.size_mode == 4) { ext = cell.size; }
    var s = p.scale * mix(1.0, sample_scale_map(cell.centre).r, p.scale_map_multiplier);
    s *= max(1.0 - p.scale_random * ff_random(cell, 13.0), 0.0);
    let turns = p.rotation + sample_rotation_map(cell.centre).r * p.rotation_map_multiplier
        + (ff_random(cell, 14.0) - 0.5) * 2.0 * p.rotation_random;
    let a = turns * 2.0 * PI;
    if (p.rotation_autoscale == 1) { s /= abs(cos(a)) + abs(sin(a)); }
    ext *= vec2f(p.size, 1.0) * max(s, 1e-6);

    // Position in the pattern.
    var rel = fx_uv(c) - cell.centre;
    rel -= round(rel);
    let off = vec2f(p.position_offset)
        + (vec2f(ff_random(cell, 15.0), ff_random(cell, 16.0)) - 0.5) * p.position_offset_random;
    if (p.position_offset_alignment == 0) { rel -= off; }
    var local = fx_rot(rel, -a) / ext + 0.5;
    if (p.position_offset_alignment == 1) { local -= off; }
    if (p.tiling_mode == 1) { local = fract(local); }
    else if (any(local < vec2f(0.0)) || any(local > vec2f(1.0))) { store(c, bg); return; }
    var v = pattern(k, local);

    if (OUT_OUTPUT_GRAY) {
        var g = v.r;
        let range = p.luminance_range * (1.0 - p.luminance_range_random * ff_random(cell, 17.0));
        g = 0.5 + (g - 0.5) * range;
        g += p.luminance_offset + (ff_random(cell, 18.0) - 0.5) * 2.0 * p.luminance_offset_random
            + sample_luminance_offset_map(cell.centre).r * p.luminance_offset_map_multiplier;
        store(c, fx_gray(g));
        return;
    }
    if (p.is_normal_map == 1) {
        let dx = p.normal_format == 0;
        let nv = fx_decode_normal(v, dx);
        // The pattern turned by a in the image (y down) turns its vectors by -a in normal space (y up).
        v = vec4f(fx_encode_normal(vec3f(fx_rot(nv.xy, -a), nv.z), dx), v.a);
    } else {
        var hsl = rgb_to_hsl(clamp(v.rgb, vec3f(0.0), vec3f(1.0)));
        hsl.x = fract(hsl.x + p.hsl_adjustment + (ff_random(cell, 19.0) - 0.5) * 2.0 * p.hsl_random);
        v = vec4f(hsl_to_rgb(hsl), v.a);
    }
    let alpha = clamp(v.a + p.alpha_adjustment + (ff_random(cell, 20.0) - 0.5) * 2.0 * p.alpha_random, 0.0, 1.0);
    store(c, vec4f(mix(bg.rgb, v.rgb, alpha), max(bg.a, alpha)));
}
