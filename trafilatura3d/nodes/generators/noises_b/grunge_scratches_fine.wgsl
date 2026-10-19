@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    let a = sat(p.scratches_amount);
    // A height map: flat mid gray, long hair-thin scratches gently bent by a low warp, raised
    // (bright) or cut (dark); Scratch Value Bias moves the balance towards the bright ones.
    let q = uv + 0.05 * warp_vec(uv, 2, 2, s + 9.0);
    let up = max(scratches(q, 4, 8, a, 0.0, 1.0, 1.8, 0.004, 0.2, s),
                 scratches(q, 8, 4, a, 0.0, 1.0, 1.8, 0.007, 0.2, s + 3.0));
    let down = max(scratches(q, 4, 6, a, 0.0, 1.0, 1.8, 0.004, 0.2, s + 6.0),
                   scratches(q, 8, 3, a, 0.0, 1.0, 1.8, 0.007, 0.2, s + 7.0));
    let vb = sat(p.scratch_value_bias);
    var v = 0.5 + 0.012 * gfbm(uv, 4, 3, 0.5, s + 10.0) + 0.2 * (0.4 + 0.6 * vb) * up - 0.16 * (1.0 - 0.6 * vb) * down;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
