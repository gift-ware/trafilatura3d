// MatFinish filters: periodic patterns (whole number of cells or waves across the UV square, so the finish tiles)
// and the relief they add to the material. Each finish defines `fn pattern(x: vec2f) -> vec4f` (x: UV, or the
// projected position in triplanar mode; .x = relief height, .yzw = the finish's own masks) and includes the
// mapping (`uvmap` or `triplanar`) that defines `mf_sample(c)` and `mf_robust()`.

fn mf_cells(n: f32) -> i32 { return max(1, i32(round(n))); }
fn mf_hash(q: vec2i, n: vec2i, s: f32) -> f32 { return hash21(wrap(q, n), s); }

// Value noise with `n` cells across the unit square, about 0..1.
fn mf_noise(x: vec2f, n: vec2i, s: f32) -> f32 {
    let y = x * vec2f(n);
    let i = vec2i(floor(y));
    let f = fract(y);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mf_hash(i, n, s), mf_hash(i + vec2i(1, 0), n, s), u.x),
               mix(mf_hash(i + vec2i(0, 1), n, s), mf_hash(i + vec2i(1, 1), n, s), u.x), u.y);
}
// Fractal value noise: `oct` octaves from `n` cells, each x2 cells and x`gain` amplitude, about 0..1.
fn mf_fbm(x: vec2f, n: i32, oct: i32, gain: f32, s: f32) -> f32 {
    var sum = 0.0; var w = 1.0; var tot = 0.0; var k = n;
    for (var o = 0; o < oct; o++) {
        sum += w * mf_noise(x, vec2i(k), s + f32(o) * 31.7);
        tot += w; w *= gain; k *= 2;
    }
    return sum / tot;
}

// Worley cells with `n` cells across, distances measured through `m` (rotation and stretch of the cells).
struct MfCell { f1: f32, f2: f32, id: vec2i, rel: vec2f };
fn mf_voronoi(x: vec2f, n: i32, jitter: f32, m: mat2x2f, s: f32) -> MfCell {
    let y = x * f32(n);
    let i = vec2i(floor(y));
    var r = MfCell(9.0, 9.0, vec2i(0), vec2f(0.0));
    for (var j = -2; j <= 2; j++) {
        for (var k = -2; k <= 2; k++) {
            let o = vec2i(k, j);
            let q = wrap(i + o, vec2i(n));
            let pt = vec2f(i + o) + 0.5 + (hash22(q, s) - 0.5) * jitter;
            let rel = m * (y - pt);
            let d = length(rel);
            if (d < r.f1) { r.f2 = r.f1; r.f1 = d; r.id = q; r.rel = rel; }
            else if (d < r.f2) { r.f2 = d; }
        }
    }
    return r;
}
fn mf_rot(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }

// Streaks (brushing, grinding): a sum of `count` waves whose directions lie within `spread` radians of the
// normal to `angle` and whose frequencies span kmin..kmax waves per unit, each rounded to whole waves across
// the UV square (tiles at any angle). Waves finer than the output's Nyquist frequency fade out. About -1..1.
fn mf_streaks(x: vec2f, angle: f32, spread: f32, kmin: f32, kmax: f32, count: i32, s: f32) -> f32 {
    let nyq = 0.5 * min(p.size_.x, p.size_.y);
    var sum = 0.0; var tot = 0.0;
    for (var i = 0; i < count; i++) {
        let r = vec4f(hash21(vec2i(i, 1), s), hash21(vec2i(i, 2), s), hash21(vec2i(i, 3), s), hash21(vec2i(i, 4), s));
        let a = angle + 0.5 * PI + (r.x - 0.5) * spread;
        let k = round(kmin * pow(kmax / kmin, r.y) * vec2f(cos(a), sin(a)));
        let f = length(k);
        let amp = (0.5 + r.w) / pow(max(f, 1.0), 0.25);
        tot += amp * amp;
        let fade = 1.0 - smoothstep(0.35 * nyq, 0.7 * nyq, f);
        if (f > 0.0) { sum += amp * fade * cos(2.0 * PI * (dot(k, x) + r.z)); }
    }
    return sum / sqrt(0.5 * tot);
}

// The pattern at a pixel with its slope (height per UV unit, +y up as the OpenGL normal): central differences,
// or the gentler one-sided difference where the mapping says neighbours may jump (UV seams in triplanar).
struct MfRelief { v: vec4f, g: vec2f };
fn mf_relief(c: vec2i) -> MfRelief {
    let v = mf_sample(c);
    let hr = mf_sample(c + vec2i(1, 0)).x; let hl = mf_sample(c - vec2i(1, 0)).x;
    let hd = mf_sample(c + vec2i(0, 1)).x; let hu = mf_sample(c - vec2i(0, 1)).x;
    var g: vec2f;
    if (mf_robust()) {
        let gx = select(v.x - hl, hr - v.x, abs(hr - v.x) < abs(v.x - hl));
        let gy = select(hu - v.x, v.x - hd, abs(v.x - hd) < abs(hu - v.x));
        g = vec2f(gx, gy) * p.size_;
    } else {
        g = vec2f(hr - hl, hu - hd) * 0.5 * p.size_;
    }
    return MfRelief(v, g);
}
// The input normal (OpenGL) with the relief's slope `g` x `k` added on top (whiteout blend).
fn mf_normal(n_in: vec4f, g: vec2f, k: f32) -> vec4f {
    let d = normalize(vec3f(-g * k, 1.0));
    let a = normalize(n_in.xyz * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5));
    let n = normalize(vec3f(a.xy + d.xy, a.z * d.z));
    return vec4f(n * 0.5 + 0.5, n_in.a);
}
fn mf_gray(v: f32) -> vec4f { let x = clamp(v, 0.0, 1.0); return vec4f(x, x, x, 1.0); }
