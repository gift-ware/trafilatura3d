// Dirt Coarse (Legacy), ours: a white noise of one value per output pixel, smeared by a small separable kernel (5 x 5:
// weights 1, k1, k2 from the centre; p.kx1 p.kx2 along x, p.ky1 p.ky2 along y, the x ones longer: short horizontal
// dashes) normalised to its sum, then a Levels (p.low, p.high, gamma p.gamma). The pixel is the grain at every
// resolution; the kernel comes from Python per output size.

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let kx = array<f32, 5>(p.kx2, p.kx1, 1.0, p.kx1, p.kx2);
    let ky = array<f32, 5>(p.ky2, p.ky1, 1.0, p.ky1, p.ky2);
    var cols: array<i32, 5>;  // the wrapped columns, once (the noise tiles)
    for (var i = 0; i < 5; i++) { cols[i] = wrap1(c.x + i - 2, d.x); }
    var sum = 0.0;
    for (var j = 0; j < 5; j++) {
        let row = wrap1(c.y + j - 2, d.y);
        var line = 0.0;
        for (var i = 0; i < 5; i++) { line += kx[i] * hash21(vec2i(cols[i], row), p.seed_); }
        sum += ky[j] * line;
    }
    let g = sum / ((1.0 + 2.0 * (p.kx1 + p.kx2)) * (1.0 + 2.0 * (p.ky1 + p.ky2)));
    let v = pow(clamp((g - p.low) / (p.high - p.low), 0.0, 1.0), p.gamma);
    store(c, vec4f(v, v, v, 1.0));
}
