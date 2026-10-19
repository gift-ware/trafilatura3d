@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Main grunge: warped blotches with faint contour rims, over soft vertical streaks.
    let wq = uv + 0.05 * warp_vec(uv, 4, 4, s + 3.0);
    let g = cfbm(wq, 4, 6, 0.6, s);
    let rim = smoothstep(0.4, 0.5, abs(fract(g * 4.0) - 0.5));
    let streak = vnoise(vec2f(uv.x * 24.0, uv.y * 2.0), vec2i(24, 2), s + 5.0) - 0.5;
    var v = 0.56 - p.main_grunge_intensity * (0.2 * (smoothstep(0.3, 0.75, g) - 0.5) + 0.06 * rim - 0.16 * streak);
    // Scratches: thin, mostly vertical, on two densities.
    let scr = max(scratches(uv, 6, 3, 0.7, 0.25, 0.12, 1.8, 0.006, 0.3, s + 20.0),
                  0.6 * scratches(uv, 12, 2, 0.7, 0.25, 0.25, 1.8, 0.01, 0.3, s + 21.0));
    let sgn = select(-1.0, 1.0, p.invert_scratches != 0);
    v += sgn * 0.3 * p.scratches_intensity * scr;
    v += 0.4 * p.grain_intensity * ((grain(uv, 1024, s + 30.0) - 0.5) + 0.5 * (grain(uv, 512, s + 31.0) - 0.5));
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
