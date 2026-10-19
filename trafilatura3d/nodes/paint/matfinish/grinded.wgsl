// .x relief, .y marks (-1..1), .z which zone (0..1).
fn pattern(x: vec2f) -> vec4f {
    var h = 0.0; var wsum = 0.0; var zone = 0.0;
    for (var i = 0; i < 3; i++) {
        let s = p.seed_ + f32(i) * 19.0;
        let a = (hash21(vec2i(0, 7), p.seed_) + f32(i) / 3.0 + (hash21(vec2i(i, 8), p.seed_) - 0.5) * 0.15) * PI;  // spread apart
        let w = smoothstep(0.35, 0.65, mf_fbm(x, mf_cells(3.0 * p.scale), 3, 0.5, s + 5.0)) + 0.05;
        h += w * mf_streaks(x, a, 0.2, 25.0 * p.scale, 400.0 * p.scale, 48, s);
        zone += w * f32(i) / 2.0;
        wsum += w;
    }
    let v = h / wsum;
    return vec4f(v / (300.0 * max(p.scale, 0.05)), v, zone / wsum, 0.0);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let k = p.grinding_intensity;
    store_roughness(c, mf_gray(load_roughness(c).r + k * (0.06 * r.v.y + 0.25 * (r.v.z - 0.5))));
    store_normal(c, mf_normal(load_normal(c), r.g, k * 0.5));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 + k * (0.06 * r.v.y - 0.15 * (r.v.z - 0.5))), vec3f(0.0), vec3f(1.0)), bc.a);
}
