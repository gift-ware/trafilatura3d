@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Gray streaky grime in a network of warped cells with bright glowing borders, washed white by clouds.
    let wq = uv + 0.012 * warp_vec(uv, 12, 3, s + 1.0);
    let e = voronoi_edge(wq * 12.0, vec2i(12), 0.9, s);
    let border = 1.0 - smoothstep(0.03, 0.08, e);
    let glow = 1.0 - smoothstep(0.0, 0.25, e);
    let grime = afbm(uv, vec2i(24, 6), 4, 0.6, s + 2.0) + 0.7 * gfbm(uv, 64, 2, 0.5, s + 3.0);
    let wash = wfbm(uv, 3, 5, 0.55, 0.08, s + 4.0);
    var v = 0.72 + 1.1 * grime + 1.9 * wash + 0.35 * glow;
    v = max(v, border);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
