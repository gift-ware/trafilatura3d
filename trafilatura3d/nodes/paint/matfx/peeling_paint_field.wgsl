// Peel field (high = peels first): warped flake cells stretched along X / Y, a noise, small flakes, and the
// convex edges of the curvature map reached within Curvature Distance.
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let base = 3.0 + p.flaking_density * 18.0;
    let n = vec2i(max(vec2f(1.0), round(base * vec2f(1.5 - p.flaking_x_amount, 1.5 - p.flaking_y_amount))));
    let warp = vec2f(mx_fbm(uv, 4, 4, 0.5, p.seed_ + 1.0), mx_fbm(uv, 4, 4, 0.5, p.seed_ + 2.0)) - 0.5;
    let wuv = uv + warp * 0.6 / vec2f(n);
    let cl = mx_cells(wuv, n, 1.0, p.seed_ + 3.0);
    let flakes = cl.z * smoothstep(0.0, 0.12, cl.w);
    let small = mx_cells(wuv, n * 3, 1.0, p.seed_ + 4.0);
    let small_flakes = step(1.0 - p.flake_level * 0.5, small.z) * smoothstep(0.0, 0.1, small.w);
    let noise = mx_fbm(uv, 5, 5, 0.6, p.seed_ + 5.0);
    var f = noise * 0.55 + flakes * 0.45 + small_flakes * 0.4;
    if (p.use_curvature != 0) {
        var k = max(load_curvature(c).r - 0.5, 0.0);
        let r = p.curvature_distance * mx_px();
        for (var i = 0; i < 8; i++) {
            let a = f32(i) / 8.0 * 2.0 * PI;
            k = max(k, max(sample_curvature(uv + vec2f(cos(a), sin(a)) * r / p.size_).r - 0.5, 0.0) * 0.8);
        }
        f += mx_contrast(mx_sat(k * 2.0), p.curvature_contrast) * 0.5;
    }
    store_peel(c, vec4f(vec3f(f * 0.8), 1.0));
}
