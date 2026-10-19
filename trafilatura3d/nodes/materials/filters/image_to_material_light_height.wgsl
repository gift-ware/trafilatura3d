@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let l = load_l(c).r;
    // Frequency bands of the luminance, largest first: blurs of 24, 6 and 2 px (at 256) and the average.
    let b1 = sample_b1(uv).r;
    let b3 = sample_b3(uv).r;
    let b2 = sample_b2(uv).r;
    let bands = vec4f(b2 - sample_lm(uv).r, b3 - b2, b1 - b3, l - b1);
    let fine = bands.w;
    var h = 0.5 + p.depth_balance * 0.5 + 1.5 * dot(bands, p.relief_equalizer) + fine * p.pinch_intensity * 2.0;
    store(c, g4(sat(h)));
}
