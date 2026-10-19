@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Horizontal bands run down in vertical streaks: the band noise is read at a height shifted per
    // column (jagged edges, smears), plus a fine column streak.
    let shift = 0.12 * pow(sat(0.5 + afbm(uv, vec2i(64, 1), 3, 0.6, s + 1.0)), 3.0) + 0.03 * afbm(uv, vec2i(12, 1), 2, 0.5, s + 2.0);
    let q = vec2f(uv.x, uv.y + shift);
    let band = afbm(q, vec2i(1, 3), 5, 0.55, s) + 0.4 * afbm(q, vec2i(4, 4), 4, 0.55, s + 4.0);
    let streak = afbm(uv, vec2i(128, 3), 2, 0.5, s + 3.0);
    var v = 0.85 + 2.2 * band + 0.4 * streak + 0.3 * gfbm(q, 32, 3, 0.6, s + 5.0);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
