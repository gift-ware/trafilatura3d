// Convexity of the stack below: divergence of the normal channel plus the height channel's negative laplacian,
// over 1.5 pixels at 256 px. 0.5 = flat, above = convex (edges), below = concave (cavities).
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.5 * mx_px() / p.size_;
    let dx = vec2f(o.x, 0.0);
    let dy = vec2f(0.0, o.y);
    let div = (mx_ndec(sample_normal(uv + dx).rgb).x - mx_ndec(sample_normal(uv - dx).rgb).x)
            + (mx_ndec(sample_normal(uv - dy).rgb).y - mx_ndec(sample_normal(uv + dy).rgb).y);
    let h = sample_height(uv).r;
    let lap = sample_height(uv + dx).r + sample_height(uv - dx).r + sample_height(uv + dy).r + sample_height(uv - dy).r - 4.0 * h;
    let v = clamp(div * 0.5 - lap * 8.0, -1.0, 1.0);
    store_edges(c, vec4f(vec3f(0.5 + 0.5 * v), 1.0));
}
