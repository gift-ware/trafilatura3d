// Leaking shapes: the bright (x) and dark (y) blotches of a warped noise at uv, each in [0, 1].
fn leak_shapes(uv: vec2f, s: f32) -> vec2f {
    let wq = uv + 0.04 * warp_vec(uv, 6, 3, s + 2.0);
    let g = cfbm(wq, 3, 7, 0.65, s);
    let detail = 0.7 + 0.3 * fbm(wq, 48, 3, 0.6, s + 1.0);
    return vec2f(smoothstep(0.62, 0.72, g) * detail, 1.0 - smoothstep(0.38, 0.5, g));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Every blotch leaks down its column: the strongest shape above the pixel, within a run whose
    // length varies column by column (Drip Length), fading with the distance. Crisp drips keep
    // their value down the run, soft ones fade early.
    let col = hcell(vec2i(i32(floor(uv.x * 64.0)), 0), vec2i(64, 1), s + 13.0) * 0.5 +
              vnoise(vec2f(uv.x * 24.0, 0.5), vec2i(24, 1), s + 14.0) * 0.5;
    let run = 0.4 * p.drip_length * (0.2 + 0.8 * col);
    let fade = mix(2.5, 0.6, sat(p.drip_crispness));
    var m = vec2f(0.0);
    let n = 32;
    let jit = hash21(c, s + 16.0);
    for (var k = 0; k < n; k++) {
        let t = (f32(k) + jit) / f32(n);
        let w = pow(1.0 - t, fade);
        m = max(m, leak_shapes(vec2f(uv.x, uv.y - run * t), s) * w);
    }
    // Vertical grain along the runs.
    let streak = vnoise(vec2f(uv.x * 384.0, uv.y * 16.0), vec2i(384, 16), s + 15.0) - 0.5;
    let sc = sat(p.shape_contrast);
    var v = 0.42 + 0.75 * (0.3 + 0.7 * sc) * m.x - 0.4 * (1.0 - 0.6 * sc) * m.y + 0.05 * streak;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
