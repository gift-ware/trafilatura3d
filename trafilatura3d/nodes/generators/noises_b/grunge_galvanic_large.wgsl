@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Zinc deposit: large clouds of the coating, warped into puffy lobes whose flat terraces end
    // in soft steps (the crystals' fronts), over a fine wavy grain.
    let q = uv + p.warp_intensity * 0.06 * warp_vec(uv, 6, 4, s + 5.0);
    let f = cfbm(q, 3, 7, 0.6, s);
    let k = 7.0;
    let step_f = (floor(f * k) + smoothstep(0.75, 1.0, fract(f * k))) / k;
    var v = 0.04 + mix(f, step_f, 0.6);
    let fine = abs(fract(4.0 * gfbm(q + 0.01 * warp_vec(uv, 24, 2, s + 6.0), 24, 3, 0.5, s + 8.0)) - 0.5);
    v -= 0.08 * (smoothstep(0.4, 0.5, fine) - 0.1);
    // Ridge details: thin bright seams along the zero lines of a second warped noise, shown where
    // the coating is high, with a dark edge beside them.
    let n = gfbm(q + 0.015 * warp_vec(uv, 24, 3, s + 9.0), 10, 4, 0.55, s + 12.0);
    let seam = 1.0 - smoothstep(0.0, 0.012, abs(n));
    let edge = 1.0 - smoothstep(0.012, 0.03, abs(n - 0.02));
    let where_ = smoothstep(0.35, 0.6, f);
    v += p.ridge_detail_opacity * where_ * (1.3 * seam - 0.1 * edge);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
