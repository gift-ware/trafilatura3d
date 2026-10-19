// Directional noise 2-4: value-noise octaves and a pixel grain, each box-blurred along the direction
// (a periodic lattice sampled along a line: tiles at any angle). An octave is smeared over a few of
// its own periods, the grain over 1/32 of the tile. Per variant (tuned):
// first octave in Scale cells, octaves, smear in periods, grain share of the variance, mean, std
// of the output, and the std of the octave mean before normalization.
const F0 = array<f32, 3>(2.0, 4.0, 8.0);
const OCTAVES = array<i32, 3>(4, 4, 3);
const SMEAR = array<f32, 3>(8.0, 4.0, 8.0);
const GRAIN = array<f32, 3>(0.4, 0.4, 0.0);
const MEAN = array<f32, 3>(0.25, 0.51, 0.5);
const STD = array<f32, 3>(0.134, 0.119, 0.089);
const OCT_STD = array<f32, 3>(0.0409, 0.0616, 0.0505);
const ROUGH = 0.8;
const GRAIN_LEN = 1.0 / 32.0;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per0 = vec2i(round(f32(max(p.scale, 1)) * asp));
    let s = p.seed_ + f32(VARIANT + 1) * 31.0;
    // Tile units; Disorder slides everything by a step in cells of the Scale grid.
    let shift = disorder_dir(hseed(0, s), p.disorder_anisotropy, p.disorder_anisotropy_angle)
        * p.disorder * p.disorder_speed / f32(max(p.scale, 1));
    let q = uv_of(c) * asp + p.tile_offset + shift;
    let a = p.angle + p.angle_random * (2.0 * vnoise(q * vec2f(per0) / asp, per0, s + 5.0) - 1.0);
    let dir = rot2(vec2f(1.0, 0.0), a);
    var b = 0.0;
    var ws = 0.0;
    var w = 1.0;
    var f = F0[VARIANT];
    for (var o = 0; o < OCTAVES[VARIANT]; o++) {
        let per = vec2i(round(f32(max(p.scale, 1)) * f * asp));
        let len = SMEAR[VARIANT] / (f32(max(p.scale, 1)) * f);
        let n = i32(2.0 * SMEAR[VARIANT]);
        var acc = 0.0;
        for (var k = 0; k < n; k++) {
            let y = q + dir * len * ((f32(k) + 0.5) / f32(n) - 0.5);
            acc += vnoise(y * vec2f(per) / asp, per, s + f32(o) * 7.0);
        }
        b += w * (acc / f32(n) - 0.5);
        ws += w;
        w *= ROUGH;
        f *= 2.0;
    }
    var z = b / ws / OCT_STD[VARIANT];
    if (GRAIN[VARIANT] > 0.0) {
        // One value per pixel (lattice points on pixel centres), averaged over n pixels.
        let n = max(i32(round(GRAIN_LEN * p.size_.x)), 1);
        let px = q * p.size_ / asp - 0.5;
        var g = 0.0;
        for (var k = 0; k < n; k++) {
            g += vnoise(px + dir * (f32(k) - 0.5 * f32(n - 1)), d, s + 101.0) - 0.5;
        }
        let gz = g / f32(n) * sqrt(12.0 * f32(n));
        z = sqrt(1.0 - GRAIN[VARIANT]) * z + sqrt(GRAIN[VARIANT]) * gz;
    }
    let v = clamp(MEAN[VARIANT] + STD[VARIANT] * z, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
