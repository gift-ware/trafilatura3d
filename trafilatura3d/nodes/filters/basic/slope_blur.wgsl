// Slope Blur: from the pixel, Samples steps along the Slope map's gradient (each the move of a Warp of
// Intensity / Samples), the values met combined: Blur a gaussian-weighted mean along the path (sigma p.spread x
// the path), Min / Max the same mean of the extreme met so far (an erosion / dilation growing along the path). Intensity 0: the input.
// The slope's gradient comes from the first pass (slope_blur_gradient.wgsl), as atomic/warp.wgsl computes it.
fn slope_at(uv: vec2f) -> vec2f {
    let g = sample_gradient(uv);
    return vec2f(g.r, g.a);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var pos = uv_at(c);
    let n = clamp(p.samples, 0, 32);
    if (p.intensity == 0.0 || n == 0) { store(c, sample_input(pos)); return; }
    let step = p.intensity / f32(n) / 256.0;
    var acc = vec4f(0.0);
    var wsum = 0.0;
    var seen = vec4f(0.0);  // Min / Max: the extreme met so far along the path
    for (var k = 1; k <= n; k++) {
        pos += slope_at(pos) * step;
        let v = sample_input(pos);
        if (k == 1) { seen = v; }
        if (p.mode == 1) { seen = min(seen, v); }
        if (p.mode == 2) { seen = max(seen, v); }
        let t = f32(k) / (f32(n) * max(p.spread, 1e-3));
        let w = exp(-0.5 * t * t);
        acc += w * select(seen, v, p.mode == 0);
        wsum += w;
    }
    store(c, acc / wsum);
}
