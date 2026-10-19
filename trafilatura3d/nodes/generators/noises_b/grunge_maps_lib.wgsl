// Shared building blocks of Grunge Map 001-015 (after noise_lib.wgsl).

// Anisotropic gradient fbm, ~[-1, 1]: base period `per` cells along x and y (doubling per octave),
// so a small y period stretches the noise into vertical streaks.
fn afbm(uv: vec2f, per: vec2i, oct: i32, gain: f32, s: f32) -> f32 {
    var v = 0.0;
    var a = 1.0;
    var t = 0.0;
    var q = max(per, vec2i(1));
    for (var o = 0; o < oct; o++) {
        v += a * gnoise(uv * vec2f(q), q, s + f32(o) * 13.7);
        t += a;
        a *= gain;
        q *= 2;
    }
    return v / t;
}

// Warped gradient fbm, ~[-1, 1]: the cloudy base of most grunge maps.
fn wfbm(uv: vec2f, per: i32, oct: i32, gain: f32, warp: f32, s: f32) -> f32 {
    return gfbm(uv + warp * warp_vec(uv, per, 4, s + 31.0), per, oct, gain, s);
}
