// .x relief, .y grunge (0..1, 1 = clean), .z flake grain (0..1).
fn pattern(x: vec2f) -> vec4f {
    let ng = mf_cells(3.0 * p.scale_grunge);
    var g: f32;
    if (p.use_custom_grunge != 0 && CUSTOM_GRUNGE_CONNECTED) {
        g = sample_custom_grunge(fract(x * f32(ng))).r;
    } else {
        g = smoothstep(0.3, 0.7, mf_fbm(x, ng * 2, 6, 0.6, p.seed_ + 23.0));
    }
    let nf = mf_cells(16.0 * p.scale);
    var f: f32;
    if (p.use_custom_flake != 0 && CUSTOM_FLAKE_CONNECTED) {
        f = sample_custom_flake(fract(x * f32(nf))).r;
    } else {
        f = mf_fbm(x, nf * 8, 2, 0.6, p.seed_ + 31.0);
    }
    // Brushing bent by a periodic warp (the warp tiles, so the marks do).
    let nw = mf_cells(3.0 * p.scale);
    let w = vec2f(mf_noise(x, vec2i(nw), p.seed_ + 41.0), mf_noise(x, vec2i(nw), p.seed_ + 43.0)) - 0.5;
    let s = mf_streaks(x + w * p.brushing_warp * 0.25, 0.0, 0.08, 20.0 * p.scale, 300.0 * p.scale, 64, p.seed_);
    let h = p.brushing_intensity * s / (150.0 * max(p.scale, 0.05)) + f / (f32(nf) * 20.0);
    return vec4f(h, g, f, 0.0);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let dirt = 1.0 - r.v.y;
    let f = r.v.z - 0.5;
    store_roughness(c, mf_gray(load_roughness(c).r + 0.3 * dirt + 0.15 * f));
    store_metallic(c, mf_gray(load_metallic(c).r * (1.0 - 0.3 * dirt)));
    store_normal(c, mf_normal(load_normal(c), r.g, 0.5));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 - 0.35 * dirt + 0.1 * f), vec3f(0.0), vec3f(1.0)), bc.a);
}
