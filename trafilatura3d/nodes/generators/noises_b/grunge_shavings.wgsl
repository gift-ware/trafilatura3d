@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    let tl = max(p.scratch_spots_tiling, 1);
    let am = sat(p.scratch_spots_amount);
    // Scratched spots spread over the surface (more of them with the amount); in them tiny curled
    // shavings, bright, on two sizes, and a faint web of short fibrous scratches.
    let gather = sat(0.3 + smoothstep(0.75 - 0.5 * am, 0.95 - 0.5 * am, fbm(uv, tl, 4, 0.5, s)));
    let flakes = discs(uv + 0.004 * warp_vec(uv, tl * 32, 2, s + 2.0), tl * 16, 1, 0.35 * am, 0.08, 0.3, 0.4, s + 6.0);
    let curls = max(max(arcs(uv, tl * 12, 1, 0.3, 0.04, s + 3.0), 0.8 * arcs(uv, tl * 24, 2, 0.3, 0.06, s + 4.0)),
                    flakes.x * (0.5 + 0.5 * flakes.y));
    let fibers = scratches(uv, tl * 16, 3, 0.6, 0.0, 1.0, 0.6, 0.03, 0.2, s + 5.0);
    let dust = fbm(uv, 8, 6, 0.6, s + 7.0);
    var v = 0.12 * p.dust_intensity * smoothstep(0.5, 0.8, dust) + gather * (curls + 0.3 * fibers);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
