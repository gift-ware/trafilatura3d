// Gradient fbm stretched along y (vertical folds); Disorder turns the lattice gradients (phase shift).
fn folds(uv: vec2f, per: vec2i, s: f32) -> f32 {
    var n = 0.0;
    var a = 1.0;
    var t = 0.0;
    var q = per;
    for (var o = 0; o < 2; o++) {
        n += a * gnoise_ph(uv * vec2f(q), q, s + f32(o) * 5.0, p.disorder);
        t += a;
        a *= 0.3;
        q *= 2;
    }
    return n / t;
}

// Distance (uv) to the zero line of `folds`, from its value and finite-difference gradient.
fn fold_line(uv: vec2f, per: vec2i, s: f32) -> f32 {
    let e = 1.0 / 2048.0;
    let n = folds(uv, per, s);
    let g = vec2f(folds(uv + vec2f(e, 0.0), per, s) - n, folds(uv + vec2f(0.0, e), per, s) - n) / e;
    return abs(n) / max(length(g), 1e-3);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Wavy vertical dark stripes (zero lines of vertical folds, of varying width) over light, crinkled foil.
    let wq = uv + vec2f(0.03 * gfbm(uv, 4, 3, 0.5, s + 1.0), 0.0);
    let w = 0.003 + 0.014 * pow(fbm(uv, 4, 2, 0.5, s + 2.0), 2.0);
    let l1 = 1.0 - smoothstep(0.3 * w, 2.5 * w, fold_line(wq, vec2i(10, 1), s));
    let l2 = 1.0 - smoothstep(0.002, 0.012, fold_line(wq, vec2i(14, 2), s + 20.0));
    let foil = ridged(uv + 0.02 * warp_vec(uv, 8, 2, s + 3.0), 12, 3, 0.5, s + 4.0);
    var v = 1.05 - 0.4 * pow(foil, 3.0) + 0.4 * folds(wq, vec2i(14, 2), s + 20.0);
    v *= 1.0 - l1;
    v *= 1.0 - 0.5 * l2;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
