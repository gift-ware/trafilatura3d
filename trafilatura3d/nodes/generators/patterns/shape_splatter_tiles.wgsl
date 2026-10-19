// Shape Splatter, pass 1: one invocation per tile (p.size_ = the grid), data from the driving maps.

fn ss_bg(uv: vec2f) -> f32 { return sample_background_height(uv).r; }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let n = ss_counts();
    let cell = vec2i(id.xy);
    if (cell.x >= n.x || cell.y >= n.y) { return; }
    let nf = vec2f(n);
    let base = (vec2f(cell) + 0.5) / nf + ss_global_offset();

    // Position: random (Gaussian or uniform) and vector map, in cells.
    var off = vec2f(srnd(cell, 10u), srnd(cell, 11u));
    if (p.random_distribution == 0) { off = vec2f(grnd(cell, 12u), grnd(cell, 13u)); }
    off *= p.position_random * 0.5;
    let v0 = (sample_vector_map(base).rg - 0.5) * 2.0;
    off += v0 * p.vector_map_multiplier;
    let center = base + off / nf;

    // Background height and slope at the centre.
    let h = ss_bg(center);
    let dx = (1.0 + p.background_slope_smoothness * 8.0) / 256.0;
    let grad = vec2f(ss_bg(center + vec2f(dx, 0.0)) - ss_bg(center - vec2f(dx, 0.0)),
                     ss_bg(center + vec2f(0.0, dx)) - ss_bg(center - vec2f(0.0, dx))) / (2.0 * dx);
    let s = ss_slope(grad);
    let sl = length(s);

    // Pattern index.
    let np = clamp(p.pattern_input_number, 1, 8);
    let idx = cell.y * n.x + cell.x;
    var k = min(i32(rnd(cell, 20u) * f32(np)), np - 1);
    switch p.pattern_distribution_mode {
        case 1: { k = wrapi(idx, np); }
        case 2: { k = wrapi(cell.y, np); }
        case 3: { k = wrapi(cell.x, np); }
        default: {}
    }
    let tk = mix((f32(k) + 0.5) / f32(np), sample_pattern_distribution(center).r, p.pattern_distribution_map_multiplier);
    k = clamp(i32(floor(tk * f32(np))), 0, np - 1);

    // Rotation (turns).
    var a = f32(p.pattern_rotation) * 0.25;
    if (rnd(cell, 30u) < p.pattern_rotation_random) { a += floor(rnd(cell, 31u) * 4.0) * 0.25; }
    a += p.rotation + p.rotation_random * srnd(cell, 32u) * 0.5;
    a += sample_shape_rotation(center).r * p.rotation_map_multiplier;
    let v1 = (sample_vector_map(center).rg - 0.5) * 2.0;
    if (length(v1) > 1e-4) { a += p.vector_map_multiplier_rotation * atan2(v1.y, v1.x) / TAU; }
    if (sl > 1e-4) { a += p.rotation_from_bg_slope * min(sl * 4.0, 1.0) * atan2(s.y, s.x) / TAU; }

    // Scale and size.
    var sc = p.scale * max(1.0 + p.scale_random * srnd(cell, 40u), 0.0) * (1.0 - p.scale_no_overlap * rnd(cell, 41u));
    sc *= mix(1.0, sample_shape_scale(center).r, p.scale_map_multiplier);
    var half = ss_half0() * sc;
    half.y *= 1.0 - p.size_ratio_from_bg_slope * min(sl, 1.0) * 0.75;

    // Out of bounds: keep the tile within the gather reach.
    let hext = max(half.x * nf.x, half.y * nf.y) * 1.415;
    if (hext + max(abs(off.x), abs(off.y)) > SS_LIMIT) {
        if (p.out_of_bounds_option == 1) {
            let room = max(SS_LIMIT - hext, 0.0);
            off = clamp(off, vec2f(-room), vec2f(room));
        }
        let oc = max(abs(off.x), abs(off.y));
        if (hext + oc > SS_LIMIT) { half *= max(SS_LIMIT - oc, 0.05) / hext; sc *= max(SS_LIMIT - oc, 0.05) / hext; }
    }
    half = max(half, vec2f(1e-6));

    // Height.
    let r = 0.001 + 0.999 * rnd(cell, 50u);
    // The offset towards a random one in [-1, 1], plus the map.
    let hoff = mix(p.height_offset, srnd(cell, 51u), p.height_offset_random)
        + sample_height_offset(center).r * p.height_offset_map_multiplier;
    let hs = p.height_scale * (1.0 - p.height_scale_random * rnd(cell, 52u))
        * mix(1.0, sample_height_scale(center).r, p.height_scale_map_multiplier);
    let basez = select(hoff, h + hoff, p.height_scale_auto_adjust == 1);

    // Masking.
    var vis = rnd(cell, 60u) >= p.mask_random && sample_mask_random(center).r >= p.mask_random_map_multiplier;
    let m = p.mask_from_bg_slope;
    if ((m > 0.0 && sl > 1.0 - m) || (m < 0.0 && sl < -m)) { vis = false; }

    store_tiles_a(cell, vec4f(off / nf, a, select(-1.0, f32(k), vis)));
    store_tiles_b(cell, vec4f(half, r, basez));
    store_tiles_c(cell, vec4f(hs, h, grad));
    store_tiles_d(cell, vec4f(sc, f32(idx), 0.0, 0.0));
}
