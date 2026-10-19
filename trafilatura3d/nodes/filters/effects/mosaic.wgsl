// W1 = warp(color), Wk = warp(W(k-1)), all by the Mosaic Map at Intensity / Samples; the
// output is W(Samples). Read back from the pixel, a path of steps along the map's slope (atomic/warp.wgsl).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var q = fx_uv(c);
    let n = clamp(p.samples, 0, 16);
    let e = 1.0 / vec2f(size_mosaic_map());
    let step = p.intensity / f32(max(n, 1)) / 256.0;
    for (var i = 0; i < n; i++) {
        let g = vec2f(sample_mosaic_map(q + vec2f(e.x, 0.0)).r - sample_mosaic_map(q - vec2f(e.x, 0.0)).r,
                      sample_mosaic_map(q + vec2f(0.0, e.y)).r - sample_mosaic_map(q - vec2f(0.0, e.y)).r);
        q += g * 0.5 * vec2f(size_mosaic_map()) * step;
    }
    store(c, sample_color(q));
}
