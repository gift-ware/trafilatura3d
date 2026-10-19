@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Lumpy gray blobs on black (two disc layers, warped); where a cloud is dense they crowd and turn white.
    let wq = uv + 0.015 * warp_vec(uv, 16, 3, s + 1.0);
    let a = discs(wq, 10, 1, 0.35, 0.25, 0.55, 0.1, s);
    let b = discs(wq, 18, 1, 0.18, 0.3, 0.6, 0.1, s + 2.0);
    let shade = 0.08 * gfbm(uv, 16, 3, 0.5, s + 3.0);
    let gray = max(a.x * (0.42 + 0.2 * a.y), b.x * (0.42 + 0.2 * b.y)) + shade * max(a.x, b.x);
    let cloud = smoothstep(0.22, 0.3, wfbm(uv, 3, 4, 0.5, 0.05, s + 4.0));
    let crowd = discs(wq, 14, 2, 0.9, 0.5, 1.0, 0.15, s + 5.0).x * cloud;
    let white = max(crowd, cloud * max(a.x, b.x)) * (1.0 + shade);
    var v = max(gray, white);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
