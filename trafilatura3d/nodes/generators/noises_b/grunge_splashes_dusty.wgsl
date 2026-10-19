@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    let qn = sat(p.splashes_quantity);
    // Splashes: worm-like dried runs, the warped zero lines of a noise thickened into bands,
    // kept where a sparse mask allows (Splashes Quantity), with a mottled inside.
    let wq = uv + (0.005 + 0.02 * p.splashes_distortion) * warp_vec(uv, 24, 3, s + 2.0);
    let n = gfbm(wq, 14, 3, 0.5, s);
    let width = 0.08 * smoothstep(0.3, 0.7, fbm(uv, 24, 2, 0.5, s + 3.0));
    let band = 1.0 - smoothstep(width * 0.6, width, abs(n));
    let mask = smoothstep(0.6 - 0.25 * qn, 0.7 - 0.25 * qn, fbm(uv, 16, 3, 0.5, s + 4.0));
    let splash = band * mask * (0.65 + 0.35 * fbm(wq, 48, 3, 0.6, s + 9.0));
    // Dust: faint clouds, spreading with Dirt Spread.
    let spread = sat(p.dirt_spread);
    let dirt = smoothstep(0.6 - 0.3 * spread, 0.95 - 0.3 * spread, fbm(uv, 4, 7, 0.6, s + 6.0)) *
               (0.6 + 0.4 * grain(uv, 256, s + 7.0));
    let r = sat(p.splash_dirt_ratio);
    var v = 0.04 + 0.85 * (0.5 + 0.5 * r) * splash + 0.15 * (1.0 - 0.7 * r) * dirt;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
