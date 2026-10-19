@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // A dark band of vertically streaked grime over light mottled clouds, with fine streaks and grit.
    let band = afbm(uv, vec2i(1, 2), 2, 0.4, s);
    let drip = afbm(uv, vec2i(8, 2), 4, 0.55, s + 1.0);
    let cloud = wfbm(uv, 4, 5, 0.6, 0.08, s + 2.0);
    let streak = afbm(uv, vec2i(96, 6), 2, 0.5, s + 3.0);
    let grit = gfbm(uv, 64, 2, 0.6, s + 4.0);
    let dark = smoothstep(0.0, 0.08, band + 0.35 * drip);
    let plates = smoothstep(-0.01, 0.01, wfbm(uv, 8, 5, 0.55, 0.05, s + 5.0)) - 0.5;
    var v = 0.52 + 1.1 * cloud + 0.2 * streak + 0.5 * grit + 0.2 * plates;
    v = mix(v, 0.08 + 0.5 * (v - 0.4), dark);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
