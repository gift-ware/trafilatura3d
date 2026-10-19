@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Mostly clean: soft gray stains where a cloud is dense, sharp dark grime marks inside them.
    let cloud = wfbm(uv, 3, 5, 0.55, 0.1, s);
    let detail = afbm(uv + 0.01 * warp_vec(uv, 8, 3, s + 4.0), vec2i(16, 10), 5, 0.65, s + 2.0);
    let stain = smoothstep(0.0, 0.3, cloud + 0.12);
    let marks = max(smoothstep(0.2, 0.26, detail + 0.6 * cloud),
                    smoothstep(0.28, 0.32, afbm(uv, vec2i(48, 24), 3, 0.6, s + 5.0) + 0.3 * cloud));
    var v = 1.15 - 0.9 * stain * (0.5 + 1.5 * detail + 0.5 * gfbm(uv, 12, 3, 0.5, s + 3.0)) - 0.9 * marks;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
