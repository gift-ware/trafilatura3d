// Brush stroke stamps: r = coverage, g / b = the stroke's two random values, a = bristle relief (0.5 flat).
// In UV space (Strokes Follow Surface, or without the position map) or projected from the three world axes.

// Stroke shape at local coordinates (a along, b across, -1..1): x = alpha, y = bristle relief.
fn shape(kind: i32, a: f32, b: f32, r: f32) -> vec2f {
    let br = 0.5 + 0.5 * sin((b + 1.0) * 13.0 + r * 40.0) * (0.6 + 0.4 * sin((b + 1.0) * 31.0 + r * 17.0));
    switch kind {
        case 0: {  // Flat: a squarish stroke with bristle streaks and a ragged end
            let d = length(max(abs(vec2f(a, b)) - vec2f(0.75, 0.7), vec2f(0.0)) / vec2f(0.25, 0.3));
            let rag = smoothstep(0.6, 1.0, a) * (1.0 - br);
            return vec2f((1.0 - smoothstep(0.85, 1.0, d)) * (1.0 - rag * 0.8), br);
        }
        case 1: {  // Round: a dab, its edge wobbling
            let ang = atan2(b, a);
            let rr = 0.85 + 0.12 * sin(ang * 5.0 + r * 30.0) + 0.05 * sin(ang * 11.0 + r * 9.0);
            let d = length(vec2f(a, b)) / rr;
            return vec2f(1.0 - smoothstep(0.85, 1.0, d), 0.5 + (br - 0.5) * 0.4);
        }
        case 2: {  // Dry: flat stroke broken by the bristles along its length
            let d = length(max(abs(vec2f(a, b)) - vec2f(0.7, 0.75), vec2f(0.0)) / vec2f(0.3, 0.25));
            let dry = smoothstep(0.25, 0.6, br + 0.25 - abs(a) * 0.35);
            return vec2f((1.0 - smoothstep(0.85, 1.0, d)) * dry, br);
        }
        default: {  // Long: a thin tapered stroke
            let taper = mix(1.0, 0.35, mx_sat((a + 1.0) * 0.5));
            let d = length(vec2f(max(abs(a) - 0.8, 0.0) / 0.2, b / (0.45 * taper)));
            return vec2f(1.0 - smoothstep(0.85, 1.0, d), br);
        }
    }
}

fn stroke_kind(q: vec2i, seed: f32) -> i32 {
    if (p.strokes_mode == 1) { return p.strokes_select_single; }
    let r = hash21(q, seed + 4.0);
    if (p.strokes_select_multiple == 1) { return select(0, 1, r > 0.5); }
    if (p.strokes_select_multiple == 2) { return select(2, 3, r > 0.5); }
    return min(i32(r * 4.0), 3);
}

// Stamps of a plane coordinate `x` (cells of the grid, `n` per period): the top stroke covering it.
fn stamps(x: vec2f, n: i32, seed: f32) -> vec4f {
    let i = vec2i(floor(x));
    var best = vec4f(0.0, 0.5, 0.5, 0.5);
    var prio_best = -1.0;
    for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
            let cell = i + vec2i(k, j);
            let q = wrap(cell, vec2i(n));
            if (hash21(q, seed + 1.0) >= p.strokes_amount) { continue; }
            let prio = hash21(q, seed + 2.0);
            if (prio <= prio_best) { continue; }
            let h = hash22(q, seed);
            let ctr = vec2f(cell) + 0.5 + (h - 0.5) * 0.5;  // with s <= 0.75 a stroke stays within 1 cell
            let ang = radians(p.strokes_rotation) + (hash21(q, seed + 3.0) - 0.5) * PI * p.strokes_rotation_random;
            let dir = vec2f(cos(ang), sin(ang));
            let s = 0.75 * (1.0 - p.strokes_scale_random * hash21(q, seed + 5.0) * 0.7);
            var aspect = vec2f(1.0, 0.55);
            if (p.non_uniform_size != 0) { aspect = vec2f(1.0, mix(0.15, 1.0, p.strokes_size)); }
            let rel = x - ctr;
            let a = dot(rel, dir) / (s * aspect.x);
            let b = dot(rel, vec2f(-dir.y, dir.x)) / (s * aspect.y);
            if (abs(a) > 1.0 || abs(b) > 1.0) { continue; }
            let sh = shape(stroke_kind(q, seed), a, b, h.x);
            if (sh.x <= 0.02) { continue; }
            prio_best = prio;
            best = vec4f(sh.x, hash21(q, seed + 6.0), hash21(q, seed + 7.0), 0.5 + (sh.y - 0.5) * sh.x);
        }
    }
    return best;
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let n = max(i32(round(12.0 / p.strokes_scale)), 1);
    if (p.strokes_follow_surface != 0 || !POSITION_CONNECTED) {
        store_strokes(c, stamps(uv * f32(n), n, p.seed_));
        return;
    }
    let pos = load_position(c).rgb * f32(n);
    let nw = mx_ndec(load_world_space_normals(c).rgb);
    var w = mx_tri(nw, p.projection_hardness) * step(vec3f(p.normal_threshold), abs(nw));
    let tot = w.x + w.y + w.z;
    if (tot < 1e-5) { store_strokes(c, vec4f(0.0, 0.5, 0.5, 0.5)); return; }
    w /= tot;
    let sx = stamps(pos.zy, n, p.seed_ + 11.0);
    let sy = stamps(pos.xz, n, p.seed_ + 22.0);
    let sz = stamps(pos.xy, n, p.seed_ + 33.0);
    var r = sx * w.x + sy * w.y + sz * w.z;
    // the random values of the axis that weighs most (blending ids makes no sense)
    if (w.x >= w.y && w.x >= w.z) { r = vec4f(r.x, sx.yz, r.w); }
    else if (w.y >= w.z) { r = vec4f(r.x, sy.yz, r.w); }
    else { r = vec4f(r.x, sz.yz, r.w); }
    store_strokes(c, r);
}
