// Tile Generator, gathered per pixel; tiles numbered row by row (tile number = row * X Amount + column).
fn tgen_counts() -> vec2i { return max(vec2i(p.x_amount, p.y_amount), vec2i(1)); }

// Size of an unscaled tile per Size Mode for lines `w` wide (uv, after Middle Size), with the per-tile randoms
// `u` (interstice) and `v` (size): in uv, or in pixels for Pixel.
fn tgen_size(w: vec2f, u: vec2f, v: vec2f) -> vec2f {
    let rs = mix(vec2f(1.0), v, p.size_random);
    switch p.size_mode {
        case 0: {                                                                       // Normal - Interstice
            let gap = p.interstice_x_y.xz * mix(vec2f(1.0), u, p.interstice_x_y.yw);
            return w * max(1.0 - gap, vec2f(0.0));
        }
        case 1: { return w * p.size * rs; }                                             // Normal - Size
        case 2: {                                                                       // Keep Ratio
            let n = vec2f(tgen_counts());
            return w * n / max(n.x, n.y) * p.size * rs;
        }
        case 3: { return p.size_absolute * rs; }                                        // Absolute
        default: { return p.size_pixel * rs; }                                         // Pixel
    }
}
// Half extent in uv of a tile of size `s` (tgen_size times its factors): Pixel rounds up to whole pixels.
fn tgen_half(s: vec2f) -> vec2f {
    if (p.size_mode == 4) { return 0.5 * ceil(s) * pixel_uv(); }
    return 0.5 * s;
}

// Middle Size along one axis of `n` cells: (width of the even lines, of the odd lines) in uv. The side set by
// Middle Size gets 2 * Middle Size cells, the other side the rest of the image (lines stay packed). Absolute and
// Pixel ignore it (0.5 there).
fn tg_middle(m: f32, n: f32) -> vec2f {
    let even = ceil(n * 0.5);
    let odd = floor(n * 0.5);
    var we = 2.0 * m / n;
    if (m > 0.5) { we = (1.0 - (1.0 - m) * 2.0 * odd / n) / max(even, 1.0); }
    return vec2f(we, (1.0 - we * even) / max(odd, 1.0));
}
fn tgen_middle() -> vec2f { return select(p.middle_size, vec2f(0.5), p.size_mode >= 3); }

// Offset of a line, in cells: Offset per line plus a random shift of up to 0.55 x Offset Random of the image
// (the same spread for every line; fitted on the output).
fn tg_line_shift(line: i32) -> f32 {
    let n = vec2f(tgen_counts());
    let cells = select(n.x, n.y, p.vertical_offset == 1);
    let r = rnd1(line, 7000u + u32(p.offset_random_seed));
    return p.offset * f32(line) + p.offset_random * r * 0.55 * cells;
}

fn tg_tile(cell: vec2i, base: vec2f) -> Tile {
    var t: Tile;
    let n = tgen_counts();
    let nf = vec2f(n);
    let idx = cell.y * n.x + cell.x;
    let odd = vec2<bool>((cell.x & 1) == 1, (cell.y & 1) == 1);
    // Masks (Horizontal hides the odd rows, Vertical the even columns).
    var hidden = (p.checker_mask == 1 && ((cell.x + cell.y) & 1) == 1)
        || (p.horizontal_mask == 1 && odd.y)
        || (p.vertical_mask == 1 && !odd.x)
        || rnd(cell, 11u) < p.random_mask;
    if (p.invert_mask == 1) { hidden = !hidden; }
    t.visible = !hidden;
    // Middle Size: line widths alternate, tiles move to the centre of their line.
    let ms = tgen_middle();
    let wx = tg_middle(ms.x, nf.x);
    let wy = tg_middle(ms.y, nf.y);
    let w = vec2f(select(wx.x, wx.y, odd.x), select(wy.x, wy.y, odd.y));
    let cf = vec2f(cell);
    let start = vec2f(ceil(cf.x * 0.5) * wx.x + floor(cf.x * 0.5) * wx.y,
                      ceil(cf.y * 0.5) * wy.x + floor(cf.y * 0.5) * wy.y);
    let middle_shift = start + 0.5 * w - (cf + 0.5) / nf;
    // Size: mode (with its randoms), scale and its random.
    let srand = 1.0 - p.scale_random * rnd(cell, 3000u + u32(p.scale_random_seed));
    // (the randoms hashed only when used: every pixel builds the tiles of 9-25 cells)
    var u = vec2f(1.0);
    if (any(p.interstice_x_y.yw > vec2f(0.0))) { u = vec2f(rnd(cell, 71u), rnd(cell, 72u)); }
    var v = vec2f(1.0);
    if (any(p.size_random > vec2f(0.0))) { v = vec2f(rnd(cell, 73u), rnd(cell, 74u)); }
    let s = tgen_size(w, u, v);
    var half = max(tgen_half(s * p.scale * srand), vec2f(1e-6));
    // Rotation in quarter turns (fixed, random, quincunx), then free rotation (fixed + random).
    var quarters = p.rotation_step;
    if (rnd(cell, 21u) < p.rotation_step_random) { quarters += i32(floor(rnd(cell, 22u) * 4.0)); }
    if (p.quincunx_flip == 1) {
        quarters += select(0, 1, odd.x) + select(0, 2, odd.y);  // a quarter turn more per step of the 2 x 2 block
    }
    if ((quarters & 1) == 1) { half = half.yx; }  // a quarter turn keeps the tile's footprint
    let a = f32(quarters) * 0.25 + p.rotation + p.rotation_random * srnd(cell, 23u);
    // Random mirroring, each axis of the mode with probability Symmetry Random / 2.
    let mirror = vec2<bool>(rnd(cell, 31u) < 0.5 * p.symmetry_random && p.symmetry_random_mode != 2,
                            rnd(cell, 32u) < 0.5 * p.symmetry_random && p.symmetry_random_mode != 1);
    let flip = select(vec2f(1.0), vec2f(-1.0), mirror);
    t.inv = mat2x2f(flip.x / half.x, 0.0, 0.0, flip.y / half.y) * rot(-a);
    t.center = middle_shift + p.position_random * vec2f(srnd(cell, 41u), srnd(cell, 42u)) / nf;
    // Colour: by number ((number + 1) / tiles) and by scale, then the random part.
    let byn = f32(idx + 1) / f32(n.x * n.y);
    if (OUT_OUTPUT_GRAY) {
        // Gray: the luminance minus the background colour (Add Sub signed, Add and Max clamped to 0-1).
        var lum = luminance(p.color.rgb);
        if (p.luminance_by_number == 1) { lum *= byn; }
        if (p.luminance_by_scale == 1) { lum *= srand; }
        lum = mix(lum, rnd(cell, 51u), p.luminance_random);
        var o = lum - luminance(p.background_color.rgb);
        if (p.blending_mode == 0 || p.blending_mode == 1) { o = clamp(o, 0.0, 1.0); }
        if (p.blending_mode == 3) { o = lum; }
        t.color = vec4f(vec3f(o), p.global_opacity);
    } else {
        var c = p.color;
        // By Number / By Scale multiply the hue, saturation, lightness or alpha of their parametrisation.
        if (p.luminance_by_number == 1) { c = hsla_scale(c, p.by_number_parametrisation, byn); }
        if (p.luminance_by_scale == 1) { c = hsla_scale(c, p.by_scale_parametrisation, srand); }
        c = color_random(c, p.color_random, cell, 52u);
        t.color = vec4f(c.rgb, c.a * p.global_opacity);
    }
    // Pattern.
    t.kind = p.pattern - 1;
    if (p.pattern == 0) {
        let m = max(p.pattern_input_number, 1);
        var k = wrapi(idx, m);
        if (p.pattern_input_distribution == 0) { k = min(i32(rnd(cell, 61u) * f32(m)), m - 1); }
        t.kind = -k - 1;
    }
    t.spec = p.pattern_specific.x;
    if (p.pattern_specific.y > 0.0) { t.spec *= mix(1.0, rnd(cell, 62u), p.pattern_specific.y); }
    t.key = f32(idx);
    return t;
}

// Premultiplied contribution (rgb) and coverage (a).
fn tg_eval(t: Tile, q: vec2f, d: vec2f) -> vec4f {
    let w = tg_aa(t);
    let op = t.color.a;
    if (t.kind >= 0) {
        let s = pattern_shape(t.kind, q, t.spec);
        return vec4f(s * t.color.rgb * op, s * op);
    }
    let s = pat_sample(-t.kind - 1, q * 0.5 + 0.5, p.image_input_filtering, w * 0.5);
    if (OUT_OUTPUT_GRAY) { return vec4f(s.rrr * t.color.rgb * op, s.r * op); }
    return vec4f(s.a * s.rgb * t.color.rgb * op, s.a * op);
}

fn tg_fold(acc: vec4f, v: vec4f) -> vec4f {
    switch p.blending_mode {
        case 1: { return vec4f(max(acc.rgb, v.rgb), acc.a); }                                    // Max
        case 3: { return vec4f(acc.rgb * (1.0 - v.a) + v.rgb, v.a + acc.a * (1.0 - v.a)); }      // Alpha Blend
        default: { return vec4f(acc.rgb + v.rgb, acc.a); }                                       // Add, Add Sub
    }
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion) - p.global_offset;
    var bg = p.background_color;
    if (OUT_OUTPUT_GRAY) { bg = vec4f(vec3f(luminance(bg.rgb)), 1.0); }
    if (BACKGROUND_CONNECTED) { bg = load_background(c); }
    let n = vec2f(tgen_counts());
    let pr = max(p.position_random.x, p.position_random.y);
    let ms = tgen_middle();
    let mid = max(abs(ms.x - 0.5), abs(ms.y - 0.5)) * 2.0;
    let h = tgen_half(tgen_size(1.0 / n, vec2f(0.0), vec2f(1.0)) * p.scale) * n;
    let ext = max(h.x, h.y) * (1.0 + mid) * 1.415 + pr + mid;
    let o = clamp(tg_render(uv, tgen_counts(), p.vertical_offset == 1, tg_reach(ext), p.blending_mode == 3,
                            p.reverse_rendering_order == 1, bg), vec4f(0.0), vec4f(1.0));
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(o.r, o.r, o.r, 1.0)); } else { store(c, o); }
}
