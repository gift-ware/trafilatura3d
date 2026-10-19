@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Clean white surface: faint thin scratches everywhere, light smudges, clusters of dark specks.
    let cluster = smoothstep(0.1, 0.35, wfbm(uv, 8, 4, 0.6, 0.05, s));
    let specks = smoothstep(0.36, 0.4, gfbm(uv, 128, 2, 0.5, s + 1.0) + 0.35 * cluster - 0.12);
    let scr = scratches(uv, 8, 2, 0.5, 0.0, 1.0, 1.6, 0.012, 0.3, s + 2.0);
    let smudge = smoothstep(0.25, 0.45, afbm(uv, vec2i(8, 16), 3, 0.5, s + 3.0)) * smoothstep(0.0, 0.2, gfbm(uv, 64, 2, 0.5, s + 4.0));
    var v = 1.0 - specks - 0.8 * scr - 0.4 * smudge;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
