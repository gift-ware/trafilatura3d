// .x relief, .y distance to the dent's centre (0 bottom .. ~1 ridge).
fn pattern(x: vec2f) -> vec4f {
    let n = mf_cells(12.0 * p.scale);
    // Distances measured in a frame turned by Rotation and stretched across it: oval dents.
    let m = mat2x2f(1.0, 0.0, 0.0, 1.3) * mf_rot(radians(p.rotation));
    let cell = mf_voronoi(x, n, 0.8, m, p.seed_);
    let d = cell.f1;
    let fine = mf_fbm(x, n * 8, 2, 0.5, p.seed_ + 3.0);
    return vec4f((d * d * 0.5 + fine * 0.02) / f32(n), clamp(d / 0.75, 0.0, 1.0), 0.0, 0.0);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let k = p.details_intensity;
    let ridge = r.v.y - 0.5;
    store_roughness(c, mf_gray(load_roughness(c).r + k * 0.15 * ridge));
    store_normal(c, mf_normal(load_normal(c), r.g, k * 0.8));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 - k * 0.08 * ridge), vec3f(0.0), vec3f(1.0)), bc.a);
}
