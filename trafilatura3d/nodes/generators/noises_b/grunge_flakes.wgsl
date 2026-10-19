@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Contrasted clouds with crisp light flakes scattered over the darker half.
    let n = wfbm(uv, 4, 6, 0.6, 0.06, s);
    let flake_n = wfbm(uv, 32, 3, 0.6, 0.01, s + 2.0);
    let flakes = smoothstep(0.17, 0.21, flake_n) * smoothstep(0.0, 0.2, gfbm(uv, 5, 3, 0.5, s + 3.0) + 0.1);
    let plates = smoothstep(-0.01, 0.01, wfbm(uv, 8, 5, 0.55, 0.05, s + 5.0)) - 0.5;
    var v = 0.6 + 2.4 * n + 0.2 * plates;
    v = max(v, flakes * (0.75 + 0.25 * fbm(uv, 32, 2, 0.5, s + 4.0)));
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
