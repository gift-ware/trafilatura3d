// Splatter Circular (ours): ring n (0 = the outer one) has radius (R - n * Ring Radius Multiplier) / R * Radius; its
// patterns sit at equally spaced angles along an arc of Spread turns centred on the left (the documentation's
// thumbnail puts the first pattern on angle 0 at Spread 1, so the whole ring turns back half a step), each moved by
// its own uniform jitter of up to SC_JITTER x Angle Random steps either way. Every pattern's position is computed directly (and its
// predecessor's, for Connect Patterns) and gathered per pixel.

// The angle jitter's half range in steps per unit of Angle Random (fitted on the output: the spread of the gaps
// between neighbours).
const SC_JITTER = 0.235;

fn sc_rings() -> i32 { return clamp(p.ring_amount, 1, 10); }
fn sc_polar(a: f32, r: f32) -> vec2f { return r * vec2f(cos(a), sin(a)); }

struct Ring {
    amount: f32,       // fractional pattern amount: the angle step is the arc / amount, the count its integer part
    count: i32,
    inv: f32,          // the inverted ring number, R - n * Ring Radius Multiplier
    radius: f32,
    first_radius: f32, // the radius of the arc's start and end (Connect Patterns)
    step: f32,         // radians
    first_angle: f32,  // pattern 0's angle before its jitter
    last_angle: f32,   // the end of the arc, where the last connected pattern closes
    first_pos: vec2f,  // where the first connected pattern starts
}

fn sc_ring(n: i32) -> Ring {
    var g: Ring;
    let rings = f32(sc_rings());
    let na = f32(max(p.pattern_amount, 1));
    g.amount = mix(na, max(rnd1(n, 900u) * na, f32(p.pattern_amount_random_min)), p.pattern_amount_random);
    g.count = i32(g.amount);
    g.inv = rings - f32(n) * p.ring_radius_multiplier;
    g.radius = g.inv * p.radius / rings;
    g.first_radius = g.radius * (1.0 - p.radius_random + p.radius_random * rnd1(n, 901u));
    let arc = p.spread * TAU;
    g.step = arc / g.amount;
    let start = (TAU - arc - g.step) * 0.5;
    let spin = TAU * (p.ring_rotation + p.ring_rotation_random * (2.0 * rnd1(n, 902u) - 1.0)
        + f32(n) / g.amount * p.ring_rotation_offset);
    let connect = p.connect_patterns == 1;
    g.first_angle = start + g.step * select(0.5, 1.0, connect) + spin;
    g.last_angle = start + arc + spin;
    g.first_pos = sc_polar(start + select(0.5 * g.step, 0.0, connect) + spin, g.first_radius * (1.0 - p.spiral_factor));
    return g;
}

// Upper bound of the distance from a pattern's placement point to its farthest pixel; `chord` = the
// distance to the previous pattern (Connect Patterns).
fn sc_reach(chord: f32) -> f32 {
    var s = max(p.size.x, p.size.y);
    if (p.connect_patterns == 1) { s = max(chord * p.size_connected.x, p.size_connected.y); }
    // Half diagonal, plus up to the size again when the pattern turns around a pivot on its edge.
    return max(p.scale, 0.0) * s * select(0.7072, 1.7072, p.pattern_rotation_pivot != 0) + p.offset_along_direction;
}

// `v`: premultiplied colour and coverage.
fn sc_fold(acc: vec3f, v: vec4f) -> vec3f {
    switch p.blending_mode {
        case 1: { return max(acc, v.rgb); }                  // Max
        case 3: { return acc * (1.0 - v.a) + v.rgb; }        // Alpha Blend
        default: { return acc + v.rgb; }  // Add and Add Sub: the same signed addition (patterns are never negative)
    }
}

// Contribution of pattern k of ring n at offset d from the layout centre: at `pos` (angle `theta`), `prev` =
// the previous pattern's position, `end` = where a connected pattern ends.
fn sc_pattern(g: Ring, n: i32, k: i32, theta: f32, pos: vec2f, prev: vec2f, end: vec2f, d: vec2f) -> vec4f {
    let cell = vec2i(k, n);
    if (rnd(cell, 11u) < p.random_mask) { return vec4f(0.0); }
    let connect = p.connect_patterns == 1;
    let rings = f32(sc_rings());
    let nk = f32(g.count);
    // Size: absolute, or the chord to the previous pattern x Size (Connected); the ring and pattern factors
    // reach 0 on the first (or, inverted, the last) one.
    var size = p.size;
    if (connect) { size = vec2f(distance(end, prev), 1.0) * p.size_connected; }
    size = mix(size, vec2f(rnd(cell, 4u), rnd(cell, 5u)) * size, p.size_random);
    let scale = p.scale * (1.0 - p.scale_random + p.scale_random * rnd(cell, 3u));
    let fr = select(f32(n), g.inv, p.invert_ring_number == 1) / rings;
    let fp = select(f32(k), nk - f32(k) - 1.0, p.invert_pattern_number == 1) / nk;
    size *= scale * mix(1.0, fr, p.scale_by_ring_number) * mix(1.0, fp, p.scale_by_pattern_number);
    if (any(size <= vec2f(0.0))) { return vec4f(0.0); }
    // Orientation: along the chord, facing the centre or none; Pattern Rotation Random blends it towards a
    // random angle. Offset along Direction follows the final rotation.
    var o = select(0.0, theta + 0.25 * TAU, p.center_orientation == 1);
    if (connect) { o = atan2(end.y - prev.y, end.x - prev.x); }
    let a = mix(o, (rnd(cell, 23u) - 0.5) * TAU, p.pattern_rotation_random) + TAU * p.pattern_rotation;
    var center = sc_polar(a, p.offset_along_direction) + select(pos, (prev + end) * 0.5, connect);
    // Pivot: a point on the pattern's edge stays where it is without the extra rotation a - o.
    if (p.pattern_rotation_pivot != 0) {
        var h = 0.5 * size.y;
        var b = 0.25 * TAU;
        switch p.pattern_rotation_pivot {
            case 1: { h = -0.5 * size.x; b = 0.0; }
            case 2: { h = 0.5 * size.x; b = 0.0; }
            case 4: { h = -0.5 * size.y; }
            default: {}
        }
        center += sc_polar(o + b, h) - sc_polar(a + b, h);
    }
    // Symmetry Random: each enabled axis mirrors with probability Symmetry Random / 2.
    var flip = select(vec2f(1.0), vec2f(-1.0), vec2f(rnd(cell, 31u), rnd(cell, 32u)) < vec2f(0.5 * p.symmetry_random));
    if (p.symmetry_random_mode == 1) { flip.y = 1.0; }
    if (p.symmetry_random_mode == 2) { flip.x = 1.0; }
    let half = size * 0.5;
    let q = flip * (rot(-a / TAU) * (d - center)) / half;
    if (any(abs(q) > vec2f(1.0))) { return vec4f(0.0); }
    // Luminance (gray) or lightness (colour) factors, the random part a random fraction of it.
    let fpl = (select(f32(k), nk - f32(k) - 1.0, p.luminance_invert_pattern_number == 1) + 1.0) / nk;
    let frl = select(f32(n) + 1.0, g.inv, p.luminance_invert_ring_number == 1) / rings;
    let f = mix(1.0, fpl, p.luminance_by_pattern_number) * mix(1.0, frl, p.luminance_by_ring_number)
        * mix(1.0, scale, p.luminance_by_scale);
    var col: vec4f;
    if (OUT_OUTPUT_GRAY) {
        col = vec4f(vec3f(luminance(p.color.rgb) * f * mix(1.0, rnd(cell, 51u), p.luminance_random)), 1.0);
    } else {
        let r = p.color_random;
        col = hsla_scale(p.color, 3, f * mix(1.0, rnd(cell, 51u), r.z));
        col = color_random(col, vec4f(r.xy, 0.0, r.w), cell, 52u);
    }
    let w = pixel_uv() / min(half.x, half.y);
    // The pattern: premultiplied value and coverage.
    var v = vec3f(0.0);
    var cov = 0.0;
    if (p.pattern == 0) {
        let m = max(p.pattern_input_number, 1);
        var pk = min(i32(rnd(cell, 61u) * f32(m)), m - 1);
        if (p.pattern_input_distribution == 1) { pk = wrapi(k, m); }
        if (p.pattern_input_distribution == 2) { pk = wrapi(i32(g.inv), m); }
        let s = pat_sample(pk, q * 0.5 + 0.5, p.image_input_filtering, w * 0.5);
        cov = select(s.a, s.r, OUT_OUTPUT_GRAY);
        v = select(s.rgb * s.a, s.rrr, OUT_OUTPUT_GRAY);
    } else {
        cov = pattern_shape(p.pattern - 1, q, p.pattern_specific);
        v = vec3f(cov);
    }
    let op = col.a * p.global_opacity;
    return vec4f(v * col.rgb * op, cov * op);
}

// Pattern k's angle: its equal step plus its jitter.
fn sc_angle(g: Ring, n: i32, k: i32) -> f32 {
    return g.first_angle + g.step * (f32(k) + SC_JITTER * p.angle_random * srnd(vec2i(k, n), 2u));
}

// Pattern k's distance from the centre: Radius Random pulls it inwards, Spiral Factor towards the centre along the
// ring.
fn sc_rho(g: Ring, n: i32, k: i32) -> f32 {
    let rho = g.radius * (1.0 - p.radius_random + p.radius_random * g.radius * rnd(vec2i(k, n), 1u));
    return mix(rho, rho * f32(k) / f32(g.count), p.spiral_factor);
}

fn sc_render(d: vec2f, acc0: vec3f) -> vec3f {
    var acc = acc0;
    let connect = p.connect_patterns == 1;
    for (var n = 0; n < sc_rings(); n++) {
        let g = sc_ring(n);
        // Every placement point of the ring is within its radius.
        if (length(d) > g.radius + sc_reach(2.0 * g.radius)) { continue; }
        var prev = g.first_pos;
        for (var k = 0; k < g.count; k++) {
            let theta = sc_angle(g, n, k);
            let pos = sc_polar(theta, sc_rho(g, n, k));
            var end = pos;
            if (k + 1 == g.count) { end = sc_polar(g.last_angle, g.first_radius); }
            if (length(d - select(pos, (prev + end) * 0.5, connect)) <= sc_reach(distance(end, prev))) {
                acc = sc_fold(acc, sc_pattern(g, n, k, theta, pos, prev, end, d));
            }
            prev = pos;
        }
    }
    return acc;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dsz = size();
    let c = vec2i(id.xy);
    if (c.x >= dsz.x || c.y >= dsz.y) { return; }
    let fs = frame_size(p.non_square_expansion);
    let uv = frame(c, p.non_square_expansion);
    var bg = p.background_color;
    if (OUT_OUTPUT_GRAY) { bg = vec4f(vec3f(luminance(bg.rgb)), 1.0); }
    if (BACKGROUND_CONNECTED) { bg = load_background(c); }
    var d = uv - fs * 0.5 - p.global_offset;
    d -= round(d / fs) * fs;
    // The layout wraps around the image: also gather from the neighbouring periodic images.
    let reach = p.radius + sc_reach(2.0 * p.radius);
    var acc = bg.rgb;
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let dd = d + vec2f(f32(ox), f32(oy)) * fs;
            if (length(dd) > reach) { continue; }
            acc = sc_render(dd, acc);
        }
    }
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(acc.r, acc.r, acc.r, 1.0)); } else { store(c, vec4f(acc, bg.a)); }
}
