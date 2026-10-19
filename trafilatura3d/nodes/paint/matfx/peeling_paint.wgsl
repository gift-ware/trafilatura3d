// Peeled = the blurred field above 1 - Peeling Level (coverage removed); the paint curls up around the holes
// over Peeling Distance; air bubbles are small domes.
fn peeled_at(f: f32) -> f32 {
    let t = 1.0 - p.peeling_level;
    return smoothstep(t - 0.004, t + 0.004, f);
}
fn rim_at(f: f32) -> f32 {
    let t = 1.0 - p.peeling_level;
    return pow(smoothstep(t - 0.02 - 0.2 * p.peeling_distance, t, f), 2.0) * (1.0 - peeled_at(f));
}
fn bubbles(uv: vec2f) -> f32 {
    let n = 12 + i32(p.air_bubble_density * 36.0);
    let cl = mx_cells(uv, vec2i(n), 1.0, p.seed_ + 9.0);
    let r = 0.18 + 0.12 * fract(cl.z * 7.0);
    return select(0.0, sqrt(mx_sat(1.0 - (cl.x * cl.x) / (r * r))), cl.z < p.air_bubble_density * 0.6);
}
fn relief(uv: vec2f) -> f32 {
    let f = sample_peel(uv).r;
    return (rim_at(f) + bubbles(uv) * 0.35) * p.height_range + (1.0 - peeled_at(f)) * 0.3;
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.0 / p.size_;
    let f = sample_peel(uv).r;
    let keep = 1.0 - peeled_at(f);
    let rim = rim_at(f);
    let bc = load_basecolor(c);
    let top = mx_hsl_adjust(mx_contrast3(bc.rgb, p.contrast), p.hue_shift, p.saturation, p.luminosity);
    store_basecolor(c, vec4f(top, bc.a * keep));
    let r = load_roughness(c);
    store_roughness(c, vec4f(r.rgb, r.a * keep));
    let rv = relief(uv);
    let slope = mx_nslope(relief(uv + vec2f(o.x, 0.0)) - rv, relief(uv - vec2f(0.0, o.y)) - rv, 4.0 * mx_px() * p.normal_intensity);
    let n = load_normal(c);
    store_normal(c, vec4f(mx_nenc(mx_ndetail(mx_ndec(n.rgb), slope)), n.a * keep));
    let h = load_height(c);
    store_height(c, vec4f(vec3f(h.r + (p.height_position - 0.5) * 0.1 + (rim + bubbles(uv) * 0.35) * p.height_range * 0.05), h.a * keep));
    let ao = load_ambient_occlusion(c);
    store_ambient_occlusion(c, vec4f(vec3f(ao.r * (1.0 - rim * p.ambient_occlusion_intensity * 0.7)), ao.a * keep));
}
