@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Vertical fibres: noise stretched along y, slightly waved sideways, in dark and light columns.
    let wq = uv + vec2f(0.02 * gfbm(uv, 4, 3, 0.5, s + 1.0), 0.0);
    let fib = afbm(wq, vec2i(32, 4), 4, 0.6, s);
    let cols = afbm(wq, vec2i(5, 1), 4, 0.5, s + 2.0);
    let grit = afbm(uv, vec2i(64, 24), 2, 0.5, s + 3.0);
    var v = 0.52 + 1.4 * fib + 1.3 * cols + 0.5 * grit;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
