// Streaks along the brushing direction (Rotation: degrees counterclockwise as seen, 0 = along U); Length narrows
// their angular spread.
fn pattern(x: vec2f) -> vec4f {
    let spread = mix(0.3, 0.004, p.length);
    let s = mf_streaks(x, radians(-p.rotation), spread, 40.0 * p.scale, 900.0 * p.scale, 128, p.seed_);
    return vec4f(s / (300.0 * max(p.scale, 0.05)), s, 0.0, 0.0);
}
fn effect(c: vec2i) -> vec4f {
    let r = mf_relief(c);
    let k = p.brushing_intensity;
    let s = r.v.y;
    store_roughness(c, mf_gray(load_roughness(c).r + 0.06 * k * s));
    store_normal(c, mf_normal(load_normal(c), r.g, k * 0.25));
    let bc = load_basecolor(c);
    return vec4f(clamp(bc.rgb * (1.0 + 0.08 * k * s), vec3f(0.0), vec3f(1.0)), bc.a);
}
