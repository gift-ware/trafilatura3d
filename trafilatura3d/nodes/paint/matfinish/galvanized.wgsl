// .x relief (fine grain, or the custom flake), .y the crystal's (or the custom flake's) value, .zw the crystal's
// tilt: a planar facet per crystal, applied as a slope (no relief step at the crystal borders).
fn pattern(x: vec2f) -> vec4f {
    let n = mf_cells(5.0 * p.scale);
    let grain = mf_fbm(x, mf_cells(96.0 * p.scale), 2, 0.5, p.seed_ + 7.0);
    if (p.use_custom_flake != 0 && CUSTOM_FLAKE_CONNECTED) {
        let v = sample_custom_flake(fract(x * f32(n))).r;
        return vec4f(v / (40.0 * f32(n)) + grain / 3000.0, v, 0.0, 0.0);
    }
    let cell = mf_voronoi(x, n, 1.0, mat2x2f(1.0, 0.0, 0.0, 1.0), p.seed_);
    return vec4f(grain / 3000.0, hash21(cell.id, p.seed_ + 5.0), hash2(cell.id, p.seed_ + 3.0));
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let k = p.flakes_intensity;
    let v = r.v.y - 0.5;
    store_roughness(c, mf_gray(load_roughness(c).r + k * 0.3 * v));
    store_normal(c, mf_normal(load_normal(c), r.g * 0.6 + r.v.zw * 0.25 * k, 1.0));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 + k * 0.35 * v), vec3f(0.0), vec3f(1.0)), bc.a);
}
