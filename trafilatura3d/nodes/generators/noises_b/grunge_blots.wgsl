// Upright oval stains on a `per` lattice (one chance per cell, probability `dens`): the smallest
// normalized ellipse distance (1 = on the rim), each oval 0.5-0.8 cell wide and 0.7-1 cell tall.
fn ovals(uv: vec2f, per: vec2i, dens: f32, s: f32) -> f32 {
    let x = uv * vec2f(per);
    let i = vec2i(floor(x));
    var best = 9.0;
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            if (hcell(q, per, s + 0.5) > dens) { continue; }
            let h = hcell2(q, per, s + 1.5);
            let pt = vec2f(q) + hcell2(q, per, s);
            let r = vec2f(mix(0.25, 0.4, h.x), mix(0.35, 0.5, h.y));
            best = min(best, length((x - pt) / r));
        }
    }
    return best;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Gray grime with large dark upright oval stains (ragged edges), light patches and crisp light specks.
    let detail = gfbm(uv, 32, 3, 0.6, s + 1.0);
    let d = ovals(uv + 0.02 * warp_vec(uv, 8, 3, s + 2.0), vec2i(5, 3), 0.45, s) + 0.6 * detail;
    let stain = 1.0 - smoothstep(0.75, 1.0, d);
    let cloud = wfbm(uv, 4, 5, 0.55, 0.06, s + 3.0);
    let specks = smoothstep(0.18, 0.24, wfbm(uv, 32, 3, 0.6, 0.01, s + 4.0)) * smoothstep(-0.2, 0.1, cloud);
    var v = 0.48 + 1.1 * cloud + 0.4 * detail;
    v = max(v, specks);
    v = mix(v, 0.25 * sat(detail + 0.1), stain);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
