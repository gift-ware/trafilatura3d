// .x relief, .y scale blotches (0..1), .z fine variation.
fn pattern(x: vec2f) -> vec4f {
    let n = mf_cells(4.0 * p.scale);
    let b = smoothstep(0.42, 0.62, mf_fbm(x, n, 6, 0.55, p.seed_));
    let fine = mf_fbm(x, n * 24, 3, 0.55, p.seed_ + 9.0);
    return vec4f((fine * 0.6 + b * 2.0) / f32(n * 24), b, fine, 0.0);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let k = p.details_intensity;
    let b = r.v.y * k;
    let f = (r.v.z - 0.5) * k;
    store_roughness(c, mf_gray(load_roughness(c).r + 0.3 * b + 0.15 * f));
    store_metallic(c, mf_gray(load_metallic(c).r * (1.0 - 0.4 * b)));
    store_normal(c, mf_normal(load_normal(c), r.g, k * 0.6));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 - 0.35 * b + 0.12 * f), vec3f(0.0), vec3f(1.0)), bc.a);
}
