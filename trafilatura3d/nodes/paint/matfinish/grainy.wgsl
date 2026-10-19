// .x relief, .y grain (0..1).
fn pattern(x: vec2f) -> vec4f {
    let n = mf_cells(128.0 * p.scale);
    let g = mf_fbm(x, n, 3, 0.6, p.seed_);
    let pits = mf_voronoi(x, n / 2, 1.0, mat2x2f(1.0, 0.0, 0.0, 1.0), p.seed_ + 11.0);
    let pit = smoothstep(0.3, 0.0, pits.f1) * step(0.7, hash21(pits.id, p.seed_ + 2.0));
    let v = g - 0.6 * pit;
    return vec4f(v / f32(n), clamp(v, 0.0, 1.0), 0.0, 0.0);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let k = p.grain_intensity;
    let v = r.v.y - 0.5;
    store_roughness(c, mf_gray(load_roughness(c).r + k * 0.3 * v));
    store_normal(c, mf_normal(load_normal(c), r.g, k * 0.5));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 + k * 0.2 * v), vec3f(0.0), vec3f(1.0)), bc.a);
}
