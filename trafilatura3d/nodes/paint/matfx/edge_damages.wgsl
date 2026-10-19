// Damage mask at uv: convex edges (blurred convexity) broken by chips, plus scratches along the edges.
fn damage(uv: vec2f) -> f32 {
    let convex = mx_sat((sample_edges(uv).r - 0.5) * 8.0);
    let cl = mx_cells(uv, vec2i(24), 1.0, p.seed_);
    let chip = mx_contrast(mx_fbm(uv, 10, 5, 0.6, p.seed_ + 3.0) * 0.7 + (1.0 - cl.x) * 0.3, 0.6);
    var d = mx_scan(convex * chip, p.level, p.contrast);
    let s = mx_scratches(uv, 18, 1.2, 0.04, p.seed_ + 11.0);
    d = max(d, s * smoothstep(0.05, 0.4, convex) * p.scratches_intensity);
    return d;
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.0 / p.size_;
    let d = damage(uv);
    let dxv = damage(uv + vec2f(o.x, 0.0)) - d;
    let dyv = damage(uv - vec2f(0.0, o.y)) - d;
    let r = load_roughness(c);
    store_roughness(c, vec4f(vec3f(mix(r.r, p.damages_roughness, d)), r.a));
    // Chip walls: the damage is a hollow, its slopes go into the normal over the stack's normal.
    let k = p.damages_depth * 4.0 * mx_px();
    let n = load_normal(c);
    store_normal(c, vec4f(mx_nenc(mx_ndetail(mx_ndec(n.rgb), mx_nslope(-dxv, -dyv, k))), n.a));
    let h = load_height(c);
    store_height(c, vec4f(vec3f(h.r - d * p.damages_depth * 0.1), h.a));
}
