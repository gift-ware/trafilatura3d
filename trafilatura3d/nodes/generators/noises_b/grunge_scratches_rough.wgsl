@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    let tl = max(p.scratch_tiling, 1);
    let q = sat(p.scratch_quantity);
    let w = 0.0015 + 0.008 * sat(p.scratch_width);
    // Blur widens the soft falloff of each scratch; length + width stay within the 3x3 search.
    let bw = min(w * (1.0 + 4.0 * sat(p.scratch_blur)), 0.15);
    let len = min(0.4 + 2.0 * sat(p.scratch_length), 1.9 - 2.0 * bw * 1.5);
    var scr = max(scratches(uv, tl, 4, q, 0.0, 1.0, len, bw, 0.5, s),
                  scratches(uv, tl * 2, 3, q, 0.0, 1.0, len, bw * 1.5, 0.4, s + 1.0));
    scr = mix(scr, smoothstep(0.0, 0.6, scr), 1.0 - sat(p.scratch_blur));
    // Double: a second, slightly warped copy beside each scratch.
    let off = uv + vec2f(0.004, 0.003) + 0.004 * warp_vec(uv, tl * 4, 2, s + 2.0);
    scr = max(scr, p.double_scratch * 0.8 * scratches(off, tl, 4, q, 0.0, 1.0, len, bw * 0.7, 0.5, s));
    // Dirtiness breaks the lines, masking hides whole regions.
    let brk = smoothstep(0.42, 0.58, fbm(uv, tl * 48, 3, 0.5, s + 4.0));
    scr *= mix(1.0, brk, sat(p.scratch_dirtiness));
    let mask = smoothstep(0.35, 0.6, fbm(uv, 3, 4, 0.5, s + 5.0));
    scr *= mix(1.0, mask, sat(p.scratch_masking));
    // Damaged spots: small bright flecks gathered in patches; dust: a faint speckle in clouds.
    let st = max(p.scratch_spots_tiling, 1);
    let spots = discs(uv, st * 4, 1, 0.6, 0.06, 0.22, 0.5, s + 7.0).x *
                smoothstep(0.45, 0.65, fbm(uv, st, 3, 0.5, s + 8.0));
    let dt = max(p.dust_tiling, 1);
    let dust = smoothstep(0.5, 0.8, fbm(uv, dt, 4, 0.6, s + 9.0)) * smoothstep(0.6, 0.9, grain(uv, 512, s + 10.0));
    var v = 0.35 * p.dust_intensity * dust + 0.9 * scr + 0.9 * p.scratch_spots_intensity * spots;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
