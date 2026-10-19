@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Bright smooth clouds blown out to white, dark blotches, crisp-edged plates and patches of grit.
    let n = wfbm(uv, 4, 6, 0.55, 0.08, s);
    let plates = smoothstep(-0.01, 0.01, wfbm(uv, 8, 5, 0.55, 0.05, s + 3.0)) - 0.5;
    let grit = smoothstep(0.15, 0.3, gfbm(uv, 96, 2, 0.5, s + 2.0)) * smoothstep(0.0, 0.25, gfbm(uv, 6, 3, 0.5, s + 4.0));
    var v = 0.85 + 2.0 * n + 0.3 * plates - 0.4 * grit;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
