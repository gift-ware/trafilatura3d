// Helpers of the image filters (included after lib.wgsl).

// Output pixels per pixel of a 256 px image: distances in "pixels at 256" hold at every size.
fn px256() -> f32 { return p.size_.x / 256.0; }

// Direction of an angle in turns, in UV (x right, y down: clockwise on screen).
fn dir_of(turns: f32) -> vec2f { let a = turns * 2.0 * PI; return vec2f(cos(a), sin(a)); }

fn rot2(v: vec2f, turns: f32) -> vec2f {
    let a = turns * 2.0 * PI;
    let c = cos(a);
    let s = sin(a);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Periodic value noise: `period` lattice cells per repetition.
fn pvalue(x: vec2f, period: i32, seed: f32) -> f32 {
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let n = vec2i(period);
    let a = hash21(wrap(i, n), seed);
    let b = hash21(wrap(i + vec2i(1, 0), n), seed);
    let c = hash21(wrap(i + vec2i(0, 1), n), seed);
    let d = hash21(wrap(i + vec2i(1, 1), n), seed);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Tileable fbm over the UV square: `cells` lattice cells at the first octave, five octaves, ~0-1.
fn pfbm(uv: vec2f, cells: i32, seed: f32) -> f32 {
    var s = 0.0;
    var amp = 0.5;
    var n = cells;
    for (var k = 0; k < 5; k++) {
        s += amp * pvalue(uv * f32(n), n, seed + f32(k) * 13.7);
        n *= 2;
        amp *= 0.5;
    }
    return clamp((s / 0.96875 - 0.5) * 1.8 + 0.5, 0.0, 1.0);
}
