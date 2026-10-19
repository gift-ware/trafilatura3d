@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Dark gray grime smudged vertically, light patches, dark grit and small light flecks.
    let smudge = afbm(uv + 0.03 * warp_vec(uv, 4, 3, s + 1.0), vec2i(6, 2), 5, 0.6, s);
    let cloud = wfbm(uv, 3, 4, 0.5, 0.08, s + 2.0);
    let grit = smoothstep(0.12, 0.2, gfbm(uv, 96, 2, 0.5, s + 3.0)) * smoothstep(-0.1, 0.15, -cloud);
    let flecks = smoothstep(0.34, 0.38, afbm(uv, vec2i(96, 64), 2, 0.5, s + 4.0) + 0.2 * cloud);
    var v = 0.38 + 1.4 * smudge + 1.0 * cloud + 0.3 * gfbm(uv, 32, 3, 0.6, s + 5.0);
    v = max(v * (1.0 - 0.8 * grit), flecks);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
