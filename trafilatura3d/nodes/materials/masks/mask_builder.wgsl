// The custom stamp scattered over jittered cells (density, size and opacity variations).
fn scatter_stamp(uv: vec2f, n: i32) -> f32 {
    let x = uv * f32(n);
    let i = vec2i(floor(x));
    var v = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
            let q = wrap(i + vec2i(k, j), vec2i(n));
            if (hash21(q, p.seed_ + 31.0) > p.density) { continue; }
            let ctr = vec2f(i + vec2i(k, j)) + hash22(q, p.seed_ + 32.0);
            let s = vec2f(max(p.scatter_size * (1.0 - p.size_variation * hash21(q, p.seed_ + 33.0)), 1e-3) * 0.5);
            let local = (x - ctr) / s * 0.5 + 0.5;
            if (all(local >= vec2f(0.0)) && all(local <= vec2f(1.0))) {
                let o = 1.0 - p.opacity_variation * hash21(q, p.seed_ + 34.0);
                v = max(v, sample_scatter_input(local).r * o);
            }
        }
    }
    return v;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    var v = 0.0;
    // AO layer.
    var a = 1.0 - load_ambient_occlusion(c).r;
    if (p.ao_invert != 0) { a = 1.0 - a; }
    a = hscan(a, p.ao_range, p.ao_contrast);
    a *= mix(1.0, noise_kind(p.ao_noise_type, uv, p.ao_noise_scale, p.seed_ + 1.0) * 2.0, p.ao_noisiness);
    v += p.ao * a;
    // Curvature layer (smoothed in the first pass).
    var k = sat(sample_blurred(uv).r * (1.0 + p.level_boost * 2.0));
    k *= mix(1.0, noise_kind(p.curvature_noise_type, uv, p.curvature_noise_scale, p.seed_ + 2.0) * 2.0, p.curvature_noisiness);
    v += p.curvature * k;
    // Top/down gradient (Position, green up): white at the top unless inverted.
    var g = load_position(c).g;
    if (p.position_invert != 0) { g = 1.0 - g; }
    v += p.top_down_gradient * hscan(g, p.position_range, p.position_contrast);
    // Six world-space directions.
    let n = wsn(load_world_space_normal(c).rgb);
    let w = p.top_intensity * sat(n.y) + p.bottom_intensity * sat(-n.y) + p.front_intensity * sat(n.z)
        + p.back_intensity * sat(-n.z) + p.right_intensity * sat(n.x) + p.left_intensity * sat(-n.x);
    v += p.world_space_normal * w;
    // Grunge multiplies the layers.
    let sc = f32(p.grunge_scale) + 1.0;
    var gr = grunge(uv, sc, p.seed_ + 3.0);
    if (p.use_triplanar != 0) {
        gr = grunge_tri(load_position(c).rgb, n, p.triplanar_blending_contrast, sc, p.seed_ + 3.0);
    }
    if (p.use_custom_grunge != 0) {
        gr = mix(load_grunge_input(c).r, load_grunge_input_2(c).r, p.secondary_custom_grunge);
    }
    if (p.grunge_invert != 0) { gr = 1.0 - gr; }
    v = mix(v, v * gr * 2.0, p.grunge);
    // Scratches and scattered stamps inside the white areas.
    let cellsn = max(i32(sqrt(f32(p.amount))), 1);
    let scr = scratches(uv, cellsn, 0.5 + p.scratches_scale * 3.0, 0.4, 1.5, 0.04 + 0.04 * p.scratches_scale, p.seed_ + 4.0);
    v += p.scratches * scr * sat(v + 0.5);
    v += p.scatter * scatter_stamp(uv, max(p.scatter_scale, 1)) * sat(v + 0.5);
    var o = hscan(v, p.level, p.contrast);
    if (p.invert != 0) { o = 1.0 - o; }
    store(c, g4(o * load_mask(c).r));
}
