// Steps of the height channel, thinned to their middle: 1 where the height is halfway between the lowest and
// highest values found within 3 pixels at 256 px, and these differ.
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let h = sample_height(uv).r;
    var lo = h;
    var hi = h;
    for (var i = 0; i < 16; i++) {
        let a = f32(i) / 16.0 * 2.0 * PI;
        let rr = select(3.0, 1.5, (i & 1) == 1) * mx_px();
        let v = sample_height(uv + vec2f(cos(a), sin(a)) * rr / p.size_).r;
        lo = min(lo, v);
        hi = max(hi, v);
    }
    let range = hi - lo;
    let mid = abs(h - 0.5 * (lo + hi)) / max(range, 1e-5);
    let e = smoothstep(0.02, 0.08, range) * smoothstep(0.3, 0.1, mid);
    store_steps(c, vec4f(vec3f(e), 1.0));
}
