// Curvature of the stack below (Input Mode: the normal channel's divergence or the height's laplacian), as the
// edges to wear (r: convex, cavities with Invert Curvature, both with Combine Curvature) and the cavities (g).
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.5 * mx_px() / p.size_;
    let dx = vec2f(o.x, 0.0);
    let dy = vec2f(0.0, o.y);
    var k = 0.0;
    if (p.input_mode == 0) {
        k = 0.5 * ((mx_ndec(sample_normal(uv + dx).rgb).x - mx_ndec(sample_normal(uv - dx).rgb).x)
                 + (mx_ndec(sample_normal(uv - dy).rgb).y - mx_ndec(sample_normal(uv + dy).rgb).y));
    } else {
        let h = sample_height(uv).r;
        k = -8.0 * (sample_height(uv + dx).r + sample_height(uv - dx).r + sample_height(uv + dy).r + sample_height(uv - dy).r - 4.0 * h);
    }
    k = clamp(k * p.curvature_intensity, -1.0, 1.0);
    var e = max(k, 0.0);
    if (p.invert_curvature != 0) { e = max(-k, 0.0); }
    if (p.combine_curvature != 0) { e = abs(k); }
    store_curv(c, vec4f(e, max(-k, 0.0), 0.0, 1.0));
}
