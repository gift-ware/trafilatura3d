// Horizontally brushed fbm: each octave stretched 1:`k` along x (fine horizontal streaks).
fn streaks(uv: vec2f, per: i32, k: i32, oct: i32, s: f32) -> f32 {
    var v = 0.0;
    var a = 1.0;
    var t = 0.0;
    var q = max(per, 1);
    for (var o = 0; o < oct; o++) {
        let pv = vec2i(max(q / k, 1), q);
        v += a * vnoise(uv * vec2f(pv), pv, s + f32(o) * 13.7);
        t += a;
        a *= 0.6;
        q *= 2;
    }
    return v / t;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Dark cement gray: large cloudy variations over a brushed, horizontally streaked base.
    let large = gfbm(uv + 0.06 * warp_vec(uv, 4, 3, s + 1.0), 4, 6, 0.6, s);
    let brushed = (streaks(uv, 128, 8, 3, s + 10.0) - 0.5) * 2.0;
    let fine = (grain(uv, 1024, s + 20.0) - 0.5) * 2.0;
    var v = 0.38 + 0.76 * p.large_variation_intensity * large + 0.05 * brushed + 0.06 * p.base_noisiness * fine;
    // Dirt specks: tiny dark pores, a pixel or two wide.
    let pores = discs(uv, 128, 1, 0.5, 0.06, 0.16, 0.5, s + 30.0).x * smoothstep(0.45, 0.65, fbm(uv, 8, 3, 0.5, s + 31.0));
    let dirt = select(-1.0, 1.0, p.invert_dirt != 0);
    v += dirt * 0.25 * p.dirt_specks_opacity * pores;
    // Faint thin scratches.
    let scr = scratches(uv, 6, 1, 0.5, 0.0, 1.0, 1.6, 0.006, 0.3, s + 40.0);
    v -= 0.15 * p.scratches_opacity * scr;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
