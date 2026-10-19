// The curvature map split in seven bands of detail (differences between gaussian blurs of 1, 2, 4... 32 px at
// 256) and summed again with one weight per band (Sharp, Fine, Soft, Medium, Large, Big, Huge): all weights 1
// give the map back.
fn effect(c: vec2i) -> vec4f {
    let uv = uv_of(c);
    var b = array<f32, 7>(load_curvature(c).r, sample_b1(uv).r, sample_b2(uv).r, sample_b3(uv).r, sample_b4(uv).r,
                          sample_b5(uv).r, sample_b6(uv).r);
    var w = array<f32, 7>(p.w0, p.w1, p.w2, p.w3, p.w4, p.w5, p.w6);
    var x = 0.5 + (b[6] - 0.5) * w[6];
    for (var k = 0; k < 6; k++) { x += (b[k] - b[k + 1]) * w[k]; }
    return vec4f(vec3f(clamp(x, 0.0, 1.0)), 1.0);
}
