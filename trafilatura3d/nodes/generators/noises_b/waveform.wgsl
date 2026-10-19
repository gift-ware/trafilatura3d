// Waveform (ours): p.samples vertical bars side by side, bar k filling the column [k, k + 1) / samples. Its height is
// Amplitude max. x mix(Amplitude min., 1, F_k x (1 - Noise + Noise x a random value per bar)), F_k the signal at the
// bar's centre: a fractal sum of octaves (Scale x 2^o periods across the image, weights Roughness^o normalised, as
// many octaves as the samples resolve) of a cosine with a random phase per octave, or of a Hermite noise (random
// values and slopes on Scale x 2^o knots, periodic; Disorder x speed blends towards the next row of knots), or the
// Custom function's constant. Along its height (centred, or standing on the bottom) a bar is the Pattern
// (nodes/shapes.wgsl, before this file) on its long axis, doubled and clamped (a white core); across, it is flat. An
// output narrower than the samples averages the bars a pixel covers.

const WF_SLOPE = 3.0;     // the noise's slopes at the knots, in values per knot (fitted on the output's spectrum)
const WF_CONTRAST = 0.75; // the noise's spread around 0.5 (fitted on the output's spread)
const WF_MAX_OCTAVES = 10;

fn wf_hash(a: i32, b: i32, c: i32) -> f32 { return hash21(vec2i(a, b * 4099 + c), p.seed_ + 0.731); }

// Hermite noise of one octave at x (in knots, periodic over `knots`), row `row` of the Disorder axis.
fn wf_hermite(x: f32, knots: i32, o: i32, row: i32) -> f32 {
    let i = i32(floor(x));
    let t = x - f32(i);
    let a = wrap1(i, knots);
    let b = (a + 1) % knots;
    let va = wf_hash(a, o, 2 * row);
    let vb = wf_hash(b, o, 2 * row);
    let ga = (2.0 * wf_hash(a, o, 2 * row + 1) - 1.0) * WF_SLOPE;
    let gb = (2.0 * wf_hash(b, o, 2 * row + 1) - 1.0) * WF_SLOPE;
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * va + (t3 - 2.0 * t2 + t) * ga + (3.0 * t2 - 2.0 * t3) * vb + (t3 - t2) * gb;
}

fn wf_signal(x: f32, n: i32) -> f32 {
    if (p.function == 2) { return p.custom_function; }
    let scale = max(p.scale, 1);
    // Octaves up to the samples' resolution (two samples per period), at least one.
    var octaves = 1;
    while (octaves < WF_MAX_OCTAVES && scale * (1 << u32(octaves)) * 2 <= n) { octaves++; }
    let dis = p.disorder * p.disorder_speed;
    var sum = 0.0;
    var wsum = 0.0;
    var w = 1.0;
    for (var o = 0; o < octaves; o++) {
        let periods = scale * (1 << u32(o));
        let phase = wf_hash(o, 7, 0);
        var v = 0.0;
        if (p.function == 1) {
            v = 0.5 + 0.5 * cos(6.28318531 * (x * f32(periods) + phase));
        } else {
            let y = dis * f32(periods);
            let row = i32(floor(y));
            let ty = y - f32(row);
            let s = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0);
            let xx = (x + phase) * f32(periods);
            let h = mix(wf_hermite(xx, periods, o, row), wf_hermite(xx, periods, o, row + 1), s);
            v = clamp(0.5 + WF_CONTRAST * (h - 0.5), 0.0, 1.0);
        }
        sum += w * v;
        wsum += w;
        w *= p.roughness;
    }
    return sum / wsum;
}

fn bar_height(k: i32, n: i32) -> f32 {
    let f = wf_signal((f32(k) + 0.5) / f32(n), n);
    let r = hash21(vec2i(k, 13), p.seed_);
    let g = f * (1.0 - p.noise + p.noise * r);
    return p.amplitude_max * mix(p.amplitude_min, 1.0, g);
}

// Bar k at the row's coordinate y (uv), `px` = one pixel in uv.
fn bar(k: i32, n: i32, y: f32, px: f32) -> f32 {
    let h = bar_height(k, n);
    if (h <= 1e-5) { return 0.0; }
    let half_h = 0.5 * max(h, px);  // never thinner than a pixel
    var centre = 0.5;
    if (p.position == 1) { centre = 1.0 - half_h; }
    let q = vec2f((y - centre) / half_h, 1e-3);  // a hair off the axis, where a shape's angular terms are defined
    let s = shape(p.pattern, q, p.pattern_variation, px / half_h);
    return select(0.0, min(2.0 * s, 1.0), s > 0.0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = max(p.samples, 1);
    let y = (f32(c.y) + 0.5) / f32(d.y);
    let px = 1.0 / f32(d.y);
    var v = 0.0;
    if (d.x >= n) {
        v = bar(min(i32(floor((f32(c.x) + 0.5) * f32(n) / f32(d.x))), n - 1), n, y, px);
    } else {
        let x0 = f32(c.x) * f32(n) / f32(d.x);
        let x1 = f32(c.x + 1) * f32(n) / f32(d.x);
        for (var k = i32(floor(x0)); k < i32(ceil(x1)); k++) {
            let cover = min(x1, f32(k + 1)) - max(x0, f32(k));
            v += cover * bar(imod(k, n), n, y, px);
        }
        v /= x1 - x0;
    }
    store(c, vec4f(v, v, v, 1.0));
}
