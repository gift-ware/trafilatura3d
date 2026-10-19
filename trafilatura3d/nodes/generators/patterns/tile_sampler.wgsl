// The Tile Sampler's tiles, shared by its passes (tile_sampler_cells.wgsl computes each cell's tile once,
// tile_sampler_draw.wgsl draws them); each pass defines ts_gray(), whether the node's output is grayscale.

// Position Random's disc radius per unit (fitted on the output: coverage and spread of the scattered tiles).
const TS_DISC = 1.5;

fn ts_counts() -> vec2i { return max(vec2i(p.x_amount, p.y_amount), vec2i(1)); }
fn ts_vert() -> bool { return p.offset_type == 1 || p.offset_type == 3; }

// Image uv (0-1, wrapped) of a frame position.
fn ts_img(uv: vec2f) -> vec2f { return fract(uv / frame_size(p.non_square_expansion)); }

// Size of an unscaled tile per Size Mode: in uv, or in pixels for Pixel.
fn ts_size() -> vec2f {
    let cs = 1.0 / vec2f(ts_counts());
    switch p.size_mode {
        case 0: { return cs * p.size; }                         // Normal: in cells
        case 1: { return vec2f(min(cs.x, cs.y)) * p.size; }     // Keep Ratio: in the smaller cell side
        case 2: { return p.size_absolute; }                     // Absolute
        default: { return p.size_pixel; }                       // Pixel
    }
}
// Half extent in uv of a tile of size `s` (ts_size times its factors): Pixel rounds up to whole pixels.
fn ts_half(s: vec2f) -> vec2f {
    if (p.size_mode == 3) { return 0.5 * ceil(s) * pixel_uv(); }
    return 0.5 * s;
}

fn tg_line_shift(line: i32) -> f32 {
    if (p.offset_type >= 2) { return p.offset * f32(line); }        // global: incremental
    return p.offset * f32(line & 1);                                // quincunx: every other line
}


fn ts_mask(uv: vec2f) -> f32 {
    var m = sample_mask_map_input(ts_img(uv)).r;
    if (p.mask_map_invert == 1) { m = 1.0 - m; }
    return m;
}

// The tile of a cell, numbered row by row.
fn tg_tile(cell: vec2i, base: vec2f) -> Tile {
    var t: Tile;
    let n = ts_counts();
    let cs = 1.0 / vec2f(n);
    let idx = cell.y * n.x + cell.x;
    // Position: Position Random scatters the tile uniformly in a disc of TS_DISC x that many (larger) cells; the
    // vector and displacement maps, read at the cell's centre, move it in image units (displacement against its
    // angle).
    let ang = rnd(cell, 41u) * TAU;
    let jitter = TS_DISC * p.position_random * sqrt(rnd(cell, 42u)) * max(cs.x, cs.y) * vec2f(cos(ang), sin(ang));
    let centre = base + p.global_offset * cs;
    let pos = centre + jitter;
    let cu = ts_img(centre);
    let vm = sample_vector_map_input(cu).rg * 2.0 - 1.0;
    let dang = p.displacement_angle * TAU;
    let disp = -p.displacement_map_intensity * sample_displacement_map_input(cu).r * vec2f(cos(dang), sin(dang))
        + p.vector_map_displacement * vm;
    t.center = jitter + disp;
    // The other maps are read where the tile lands.
    let at = pos + disp;
    let fu = ts_img(at);
    // Scale: uniform + random, scale map and vector map length (on the chosen axes), size random per axis.
    let srand = mix(1.0, rnd(cell, 3u), p.scale_random);
    let sf = mix(1.0, sample_scale_map_input(fu).r, p.scale_map_multiplier) * mix(1.0, length(vm), p.scale_vector_map_multiplier);
    var mapf = vec2f(sf);
    if (p.scale_parametrization_affect == 1) { mapf.y = 1.0; }
    if (p.scale_parametrization_affect == 2) { mapf.x = 1.0; }
    let sizer = mix(vec2f(1.0), vec2f(rnd(cell, 4u), rnd(cell, 5u)), p.size_random);
    var half = max(ts_half(ts_size() * sizer * p.scale * srand * mapf), vec2f(1e-6));
    // Rotation: a 90-degree step (Rotation Random picks a random one instead) keeping the footprint, free rotation
    // +- Rotation Random turns, rotation map, vector map direction.
    var quarters = p.rotation_step;
    if (rnd(cell, 21u) < p.rotation_step_random) { quarters = i32(floor(rnd(cell, 22u) * 4.0)); }
    if ((quarters & 1) == 1) { half = half.yx; }
    var a = f32(quarters) * 0.25 + p.rotation + p.rotation_random * srnd(cell, 23u);
    a += p.rotation_map_multiplier * sample_rotation_map_input(fu).r;
    if (length(vm) > 1e-4) { a += p.vector_map_multiplier * atan2(vm.y, vm.x) / TAU; }
    // Random mirroring, each axis of the mode with probability Symmetry Random / 2.
    let mirror = vec2<bool>(rnd(cell, 31u) < 0.5 * p.symmetry_random && p.symmetry_random_mode != 2,
                            rnd(cell, 32u) < 0.5 * p.symmetry_random && p.symmetry_random_mode != 1);
    let flip = select(vec2f(1.0), vec2f(-1.0), mirror);
    t.inv = mat2x2f(flip.x / half.x, 0.0, 0.0, flip.y / half.y) * rot(-a);
    // Mask: the map at the tile (or at four corners 0.8 half-size away) and the random mask; Invert Mask flips both.
    var keep = ts_mask(at) >= p.mask_map_threshold;
    if (p.mask_map_sampling_technique == 1) {
        let e = 0.8 * max(half.x, half.y);
        keep = ts_mask(at + vec2f(e, e)) >= p.mask_map_threshold && ts_mask(at + vec2f(-e, e)) >= p.mask_map_threshold
            && ts_mask(at + vec2f(-e, -e)) >= p.mask_map_threshold && ts_mask(at + vec2f(e, -e)) >= p.mask_map_threshold;
    }
    keep = keep && rnd(cell, 11u) >= p.mask_random;
    if (p.invert_mask == 1) { keep = !keep; }
    t.visible = keep;
    // Pattern.
    let mcount = max(p.pattern_input_number, 1);
    var k = 0;
    switch p.pattern_input_distribution {
        case 1: { k = wrapi(idx, mcount); }
        case 2: { k = min(i32(sample_pattern_distribution_map_input(fu).r * f32(mcount)), mcount - 1); }
        default: { k = min(i32(rnd(cell, 61u) * f32(mcount)), mcount - 1); }
    }
    t.kind = select(p.pattern - 1, -k - 1, p.pattern == 0);
    t.spec = p.pattern_specific * mix(1.0, rnd(cell, 62u), p.pattern_specific_random);
    // Colour: parametrization first, then the random part (Luminance / Color Random); Add/Sub draws it minus the
    // background colour.
    var pv = vec4f(1.0);
    switch p.color_parametrization_mode {
        case 0, 1: { pv = sample_color_map_input(fu); }
        case 2: { pv = vec4f(vec3f(srand * sf), 1.0); }
        case 3: { pv = vec4f(vec3f(f32(cell.y + 1) / f32(n.y)), 1.0); }
        case 4: { pv = vec4f(vec3f(f32(cell.x + 1) / f32(n.x)), 1.0); }
        case 5: {
            let pi = select(f32(idx + 1) / f32(n.x * n.y), f32(k) / f32(max(mcount - 1, 1)), p.pattern == 0);
            pv = vec4f(vec3f(pi), 1.0);
        }
        case 6: { pv = vec4f(fu, 0.0, 1.0); }
        default: { pv = vec4f(fu, clamp(length(half), 0.0, 1.0), 1.0); }
    }
    let pm = p.color_parametrization_multiplier;
    var col = p.color;
    if (ts_gray()) {
        col = vec4f(vec3f(mix(mix(col.r, luminance(pv.rgb), pm), rnd(cell, 51u), p.color_luminance_random)), col.a);
    } else {
        switch p.color_parametrization_affect {
            case 1: { col = vec4f(mix(col.rgb, pv.rgb, pm), col.a); }
            case 2: { col.a = mix(col.a, luminance(pv.rgb), pm); }
            default: { col = mix(col, vec4f(pv.rgb, pv.a), pm); }
        }
        col = color_random(col, p.color_random, cell, 52u);
    }
    if (p.blending_mode == 0) { col = vec4f(col.rgb - p.background_color.rgb, col.a); }
    t.color = vec4f(col.rgb, col.a * p.global_opacity);
    t.key = f32(idx);
    return t;
}

