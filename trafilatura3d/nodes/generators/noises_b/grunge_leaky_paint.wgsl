// Flaking paint: patches of flat grays (crisp) or soft noise, at uv.
fn paint_patches(uv: vec2f, crisp: f32, s: f32) -> f32 {
    let g = cfbm(uv + 0.03 * warp_vec(uv, 8, 3, s + 2.0), 4, 7, 0.62, s);
    let flat = floor(g * 7.0) / 6.0;
    return mix(g, flat, crisp);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    let li = sat(p.leak_intensity);
    let crisp = sat(p.leak_crispness);
    // Leaks: the paint is pulled down in columns (Leak Scale columns of 24 streaks, on three
    // widths), each column by its own length, slanted by up to Leak Angle Random turns: a
    // directional warp that cuts the patches into shifted vertical strips. Dark drips run along
    // the longest pulls.
    let cols = max(p.leak_scale, 1) * 24;
    var len = 0.0;
    for (var k = 0; k < 3; k++) {
        let n = cols * (1 << u32(k)) / 2;
        let x = f32(i32(floor(uv.x * f32(n))));
        len += vnoise(vec2f(x, uv.y * 2.0), vec2i(n, 2), s + 4.0 + f32(k)) / 3.0;
    }
    let cx = i32(floor(uv.x * f32(cols)));
    let a = TAU * p.leak_angle_random * (hcell(vec2i(cx, 0), vec2i(cols, 1), s + 3.0) * 2.0 - 1.0);
    let src = uv - (0.02 + 0.16 * li) * len * vec2f(sin(a), cos(a));
    var v = 0.0 + 0.75 * paint_patches(src, crisp, s);
    v -= 0.12 * li * smoothstep(0.6, mix(0.8, 0.65, crisp), len);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
