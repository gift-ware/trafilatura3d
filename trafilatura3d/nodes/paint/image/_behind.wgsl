// An effect layer of coverage `s` and color `col` laid under the input `t` (Drop Shadow, Glow). `mask`: the input
// is a mask (its value `m` is the shape): the effect adds to it, m + (1 - m) s, as grayscale.
fn behind(t: vec4f, m: f32, s: f32, col: vec3f, mask: bool) -> vec4f {
    if (mask) { let o = m + (1.0 - m) * s; return vec4f(o, o, o, t.a); }
    let a = t.a + s * (1.0 - t.a);
    return vec4f((t.rgb * t.a + col * s * (1.0 - t.a)) / max(a, 1e-6), a);
}

// Spread chokes the effect: full coverage reached at 1 - spread.
fn spread(s: f32, k: f32) -> f32 { return clamp(s / max(1.0 - k, 1e-3), 0.0, 1.0); }
