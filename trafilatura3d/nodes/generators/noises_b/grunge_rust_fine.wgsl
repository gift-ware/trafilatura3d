@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Base: a light surface eaten by warped darker rust clouds with crinkled edges.
    let wq = uv + p.base_warp_intensity * 0.1 * warp_vec(uv, 4, 5, s + 3.0);
    let b0 = cfbm(wq, 3, 7, 0.68, s);
    let k = 0.5 + 0.8 * p.base_grunge_contrast;
    var v = 0.62 + ((b0 - 0.55) * 0.8 + 0.05 * (smoothstep(0.47, 0.52, b0) - 0.5)) * k;
    let crinkle = ridged(wq + 0.01 * warp_vec(uv, 32, 2, s + 4.0), 24, 3, 0.5, s + 5.0);
    v -= 0.08 * smoothstep(0.75, 0.95, crinkle) * smoothstep(0.6, 0.4, b0);
    // Streaks and spots: brighter, faint.
    let sy = uv.y * 4.0 + 1.5 * vnoise(vec2f(uv.x * 8.0, uv.y * 2.0), vec2i(8, 2), s + 8.0);
    let streak = vnoise(vec2f(uv.x * 96.0, sy), vec2i(96, 4), s + 9.0) * vnoise(vec2f(uv.x * 12.0, sy), vec2i(12, 4), s + 10.0);
    let spots = discs(uv, 32, 1, 0.3, 0.1, 0.35, 0.6, s + 11.0).x;
    v += 0.12 * p.streaks_intensity * max(smoothstep(0.4, 0.8, streak), 0.6 * spots);
    v += 0.15 * p.noise_intensity * ((grain(uv, 512, s + 20.0) - 0.5) + 0.5 * (grain(uv, 256, s + 21.0) - 0.5));
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
