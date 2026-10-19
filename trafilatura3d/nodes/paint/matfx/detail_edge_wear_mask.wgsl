// The wear (generator) mask: blurred edges, softened by Edges Smoothness, broken by the grunge, kept out of the
// spread cavities, leveled by Wear Level / Wear Contrast; Alpha Background Value outside the stack's coverage.
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let e = mx_sat(sample_edges(uv).r * 3.0);
    let edges = smoothstep(0.02, 0.05 + 0.7 * p.edges_smoothness, e);
    let n = max(i32(round(6.0 * p.grunge_scale)), 1);
    let g = mx_fbm(uv, n, 5, 0.6, p.seed_ + 1.0) * 0.7 + (1.0 - mx_cells(uv, vec2i(n * 3), 1.0, p.seed_ + 2.0).x) * 0.3;
    var x = edges - (1.0 - mx_contrast(g, 0.6)) * p.grunge_amount * 1.6;
    x *= 1.0 - mx_sat(sample_occlusion(uv).g * 3.0);
    x = mx_scan(x, p.wear_level, p.wear_contrast);
    x = mix(p.alpha_background_value, x, load_normal(c).a);
    store_mask(c, vec4f(vec3f(x), 1.0));
}
