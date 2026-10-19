@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Base grunge: soft dirt clouds, a web of fine dark crinkles and grain on a mid gray.
    let wq = uv + 0.05 * warp_vec(uv, 3, 3, s + 3.0);
    let dirt = gfbm(wq, 3, 6, 0.6, s);
    let crinkle = smoothstep(0.9, 0.98, ridged(wq, 48, 2, 0.5, s + 4.0)) * smoothstep(0.3, 0.6, fbm(uv, 6, 3, 0.5, s + 5.0));
    var v = 0.47 + p.base_grunge_intensity * (0.22 * dirt - 0.1 * crinkle + 0.2 * (grain(uv, 1024, s + 6.0) - 0.5));
    // Scratches in every direction, hair thin: mostly bright, some dark.
    let bright = max(scratches(uv, 16, 3, 0.7, 0.0, 1.0, 1.7, 0.006, 0.2, s + 20.0),
                     scratches(uv, 6, 2, 0.6, 0.0, 1.0, 1.7, 0.003, 0.2, s + 22.0));
    let dark = scratches(uv, 12, 2, 0.6, 0.0, 1.0, 1.7, 0.005, 0.2, s + 24.0);
    v += p.scratches_intensity * (0.2 * bright - 0.1 * dark);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
