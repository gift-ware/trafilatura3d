// Rows of the quadrants: on each side (lo: towards -x, hi: towards +x) the mean color and the mean of its
// squared length over the 4 samples.
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let d = grid_step().x;
    let m = fetch(uv).rgb;  // the center column is shared
    var lo = vec4f(m, dot(m, m));
    var hi = lo;
    for (var i = 1; i < 4; i++) {
        let a = fetch(uv - vec2f(f32(i) * d, 0.0)).rgb;
        let b = fetch(uv + vec2f(f32(i) * d, 0.0)).rgb;
        lo += vec4f(a, dot(a, a));
        hi += vec4f(b, dot(b, b));
    }
    store_lo(c, lo / 4.0);
    store_hi(c, hi / 4.0);
}
