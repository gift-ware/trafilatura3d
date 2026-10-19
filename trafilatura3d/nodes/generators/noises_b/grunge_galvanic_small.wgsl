@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Dense zinc deposit: a strongly warped noise whose level steps end in a bright ridge and a dark
    // groove (Crispness sharpens them), with small pits of concentric rings, dirt on top.
    let q = uv + (0.03 + 0.06 * p.micro_distortion) * warp_vec(uv, 12, 4, s + 5.0) * vec2f(0.5, 1.0);
    let f = cfbm(q, 10, 5, 0.6, s);
    let terr = fract(f * 4.0);
    let ridge = exp(-pow((terr - 0.92) / 0.05, 2.0)) - exp(-pow((terr - 0.06) / 0.06, 2.0));
    let flat = (floor(f * 4.0) + smoothstep(0.7, 1.0, terr)) / 4.0;
    var v = 0.05 + mix(f, flat, 0.5 * sat(p.crispness)) + 0.4 * sat(p.crispness) * ridge;
    // Pits: dark cores ringed by bright and dark contours.
    let cell = voronoi((q + 0.02 * warp_vec(uv, 24, 2, s + 6.0)) * 12.0, vec2i(12), 0.9, s + 7.0);
    let h = hash22(cell.id, s + 8.0);
    let r = mix(0.18, 0.4, h.x);
    let pit = 1.0 - smoothstep(0.85, 1.0, cell.f1 / r);
    let rings = 0.5 + 0.5 * cos(TAU * 2.5 * cell.f1 / r);
    v = mix(v, 0.15 + 0.6 * rings * smoothstep(0.2, 0.8, cell.f1 / r), 0.7 * pit * step(h.y, 0.3));
    let dirt = smoothstep(0.45, 0.75, fbm(uv, 3, 5, 0.55, s + 9.0));
    v *= 1.0 - 0.5 * p.dirt * dirt;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
