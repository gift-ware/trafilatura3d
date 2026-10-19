// Final pass: smoothing (uniform, or only on the slopes) and the normal map.
fn height(q: vec2i) -> f32 {
    let r = texel_raw(q).r;
    let b = texel_blurred(q).r;
    // A Non Uniform Blur with the (slightly blurred) ramp levelled up as its map: no blur where it is 0.
    if (p.use_non_uniform_blur == 1) { return mix(r, b, clamp(b / 0.0214, 0.0, 1.0)); }
    return b;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var h: array<f32, 9>;
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) { h[j * 3 + i] = height(c + vec2i(i - 1, j - 1)); }
    }
    let s = fx_slopes3(h, 1.0 / p.size_, true) * p.normal_intensity / 256.0;
    store_height(c, fx_gray(h[4]));
    store_normal(c, vec4f(fx_encode_normal(fx_slope_normal(s), p.normal_format == 0), 1.0));
}
