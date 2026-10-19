@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Light gray grime with dark round spots, patches of dark grit and broken fine cracks (Cracks).
    let wq = uv + 0.01 * warp_vec(uv, 16, 3, s + 1.0);
    let base = wfbm(uv, 4, 6, 0.6, 0.05, s);
    let spots = discs(wq, 20, 1, 0.45, 0.25, 0.55, 0.35, s + 2.0).x;
    let area = smoothstep(-0.2, 0.2, gfbm(uv, 4, 4, 0.5, s + 3.0));
    let grit = smoothstep(0.1, 0.2, gfbm(uv, 96, 2, 0.5, s + 4.0)) * area;
    let e = voronoi_edge(wq * 14.0, vec2i(14), 1.0, s + 5.0);
    let crack = (1.0 - smoothstep(0.025, 0.06, e)) * smoothstep(0.0, 0.15, gfbm(uv, 24, 2, 0.5, s + 6.0));
    var v = 0.82 + 1.2 * base + 1.1 * gfbm(uv, 48, 2, 0.6, s + 7.0) - 0.2 * area;
    v = v * (1.0 - spots) * (1.0 - 0.9 * grit) * (1.0 - p.cracks * crack);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
