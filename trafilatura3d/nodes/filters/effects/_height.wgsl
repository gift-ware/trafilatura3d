// Slopes of the `input` height map around output pixel c, in height per UV.
fn fx_input_slopes(c: vec2i, sobel: bool) -> vec2f {
    let uv = fx_uv(c);
    let e = 1.0 / p.size_;
    var h: array<f32, 9>;
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            h[j * 3 + i] = sample_input(uv + vec2f(f32(i - 1), f32(j - 1)) * e).r;
        }
    }
    return fx_slopes3(h, e, sobel);
}
