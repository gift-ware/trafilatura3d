// .x orange peel relief, .y flake mask (0..1), .zw the flake's tilt.
fn pattern(x: vec2f) -> vec4f {
    let n = mf_cells(20.0 * p.scale);
    let peel = mf_fbm(x, n, 2, 0.35, p.seed_);
    let nf = mf_cells(160.0 * p.scale);
    if (p.use_custom_flake != 0 && CUSTOM_FLAKE_CONNECTED) {
        let v = sample_custom_flake(fract(x * f32(mf_cells(8.0 * p.scale)))).r;
        return vec4f(peel / f32(n), v, 0.0, 0.0);
    }
    let cell = mf_voronoi(x, nf, 0.9, mat2x2f(1.0, 0.0, 0.0, 1.0), p.seed_ + 13.0);
    let flake = smoothstep(0.32, 0.2, cell.f1) * step(0.55, hash21(cell.id, p.seed_ + 17.0));
    return vec4f(peel / f32(n), flake, hash2(cell.id, p.seed_ + 19.0) * flake);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let f = r.v.y * p.flakes_intensity;
    store_roughness(c, mf_gray(load_roughness(c).r * (1.0 - 0.5 * f)));
    store_metallic(c, mf_gray(mix(load_metallic(c).r, 1.0, f)));
    store_normal(c, mf_normal(load_normal(c), r.g * 0.08 + r.v.zw * 0.5 * p.flakes_intensity, 1.0));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 + 0.4 * f), vec3f(0.0), vec3f(1.0)), bc.a);
}
