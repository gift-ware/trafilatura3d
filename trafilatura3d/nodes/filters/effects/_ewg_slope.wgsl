// Emboss With Gloss: the light at pixel c, the height's slope towards Light Angle (central differences, a
// change across one pixel at 256 px counting 1) times Intensity; positive faces the light.
fn ewg_slope(c: vec2i) -> f32 {
    let d = vec2f(size());
    let uv = (vec2f(c) + 0.5) / d;
    let e = 1.0 / d;
    let gx = sample_height(uv + vec2f(e.x, 0.0)).r - sample_height(uv - vec2f(e.x, 0.0)).r;
    let gy = sample_height(uv + vec2f(0.0, e.y)).r - sample_height(uv - vec2f(0.0, e.y)).r;
    let a = p.light_angle * 2.0 * PI;
    return -dot(vec2f(gx, gy), vec2f(cos(a), sin(a))) * p.intensity * d.x / 256.0;
}
