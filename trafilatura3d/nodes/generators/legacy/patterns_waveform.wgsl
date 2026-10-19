// Waveform (Legacy), ours: p.samples vertical bars side by side, bar k filling the column [k, k + 1) / samples, centred
// on the middle row and as tall as its height h_k = size_max x lerp(F_k, 1, size_min), F_k = W_k x (1 - noise +
// noise x a random value per bar), W_k a raised cosine of wave_number periods across the image (1 with wave_number 0).
// Along its height a bar is the shape `kind` (nodes/shapes.wgsl, after this file's prefix) on its long axis, doubled
// and clamped (a white core); across, it is flat. An output narrower than the samples averages the bars a pixel covers.

fn bar_height(k: i32, n: i32) -> f32 {
    var w = 1.0;
    if (p.wave_number != 0) {
        w = 0.5 - 0.5 * cos(6.28318531 * f32(p.wave_number) * f32(k + 1) / f32(n));
    }
    let r = hash21(vec2i(k, 13), p.seed_);
    let f = w * (1.0 - p.noise + p.noise * r);
    return p.size_max * (f * (1.0 - p.size_min) + p.size_min);
}

// Bar k at the row's coordinate y (uv), `px` = one pixel in uv.
fn bar(k: i32, n: i32, y: f32, px: f32) -> f32 {
    let h = bar_height(k, n);
    if (h <= 1e-5) { return 0.0; }  // a flat bar (the wave's zero, within float rounding) draws nothing
    let half_h = 0.5 * max(h, px);  // never thinner than a pixel
    let q = vec2f((y - 0.5) / half_h, 1e-3);  // a hair off the axis, where the angular terms of a shape are defined
    let s = shape(p.kind, q, 0.0, px / half_h);
    return select(0.0, min(2.0 * s, 1.0), s > 0.0);  // also 0 where a shape's power of a negative base is undefined
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = p.samples;
    var v = 0.0;
    if (n > 0) {
        let y = (f32(c.y) + 0.5) / f32(d.y);
        let px = 1.0 / f32(d.y);
        if (d.x >= n) {
            v = bar(min(i32(floor((f32(c.x) + 0.5) * f32(n) / f32(d.x))), n - 1), n, y, px);
        } else {
            // The bars under the pixel, each weighted by the part of the pixel it covers.
            let x0 = f32(c.x) * f32(n) / f32(d.x);
            let x1 = f32(c.x + 1) * f32(n) / f32(d.x);
            for (var k = i32(floor(x0)); k < i32(ceil(x1)); k++) {
                let cover = min(x1, f32(k + 1)) - max(x0, f32(k));
                v += cover * bar(imod(k, n), n, y, px);
            }
            v /= x1 - x0;
        }
    }
    store(c, vec4f(v, v, v, 1.0));
}
