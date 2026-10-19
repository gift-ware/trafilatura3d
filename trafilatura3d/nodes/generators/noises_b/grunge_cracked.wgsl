@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Dark speckled grime with light horizontal streaks and clouds, crossed by a wobbly crack network.
    let cloud = wfbm(uv, 3, 5, 0.55, 0.08, s);
    let streak = afbm(uv, vec2i(3, 32), 3, 0.55, s + 1.0);
    let speck = gfbm(uv, 64, 2, 0.6, s + 2.0);
    let wq = uv + 0.008 * warp_vec(uv, 24, 3, s + 3.0);
    let e = voronoi_edge(wq * 10.0, vec2i(10), 1.0, s + 4.0);
    let crack = (1.0 - smoothstep(0.03, 0.07, e)) * smoothstep(-0.1, 0.1, gfbm(uv, 16, 2, 0.5, s + 5.0));
    var v = 0.42 + 1.7 * cloud + 1.1 * streak + 0.9 * speck;
    v *= 1.0 - p.cracks * crack;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
