@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Large blotches smeared downwards: a dark and a light band, tall blotches, fine vertical streaks.
    let band = afbm(uv, vec2i(1, 2), 2, 0.4, s);
    let blot = afbm(uv, vec2i(6, 3), 5, 0.55, s + 1.0);
    let streak = afbm(uv, vec2i(64, 6), 3, 0.6, s + 2.0);
    let grit = gfbm(uv, 24, 3, 0.6, s + 3.0);
    var v = 0.6 + 1.4 * band + 1.9 * blot + 0.7 * streak + 0.4 * grit;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
