@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    let sc = max(p.scale, 1);
    // Bright dirt on black: spots (finer with Scale, more with Coverage) made of broken flecks and
    // small curls, the flecks thinning out towards the spot's edge; a few stray flecks around.
    let q = uv + 0.1 / f32(sc) * warp_vec(uv, sc * 3, 3, s + 2.0);
    let spot = discs(q, sc * 3, 2, 0.2 + 0.6 * p.coverage, 0.3, 0.7, 1.0, s).x;
    let fine = fbm(q, sc * 48, 3, 0.55, s + 5.0);
    let flecks = smoothstep(0.76 - 0.2 * spot, 0.8 - 0.2 * spot, fine) * (0.4 + 0.6 * grain(q, sc * 16, s + 7.0));
    let curls = arcs(q, sc * 16, 2, 0.4, 0.04, s + 6.0);
    var v = (0.15 + 0.85 * smoothstep(0.0, 0.3, spot)) * max(flecks, 0.6 * curls * spot);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
