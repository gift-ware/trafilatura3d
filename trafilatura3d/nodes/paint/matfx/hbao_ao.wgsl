// Horizon based occlusion: 8 jittered directions, 10 steps up to Radius; each rise of the horizon's sine
// occludes, weighted by the distance falloff (Relief Balance: steep near falloff at 0, flat at 1).
fn hgt(uv: vec2f) -> f32 {
    var h = sample_height(uv).r;
    if (p.channel_source == 1) { h = mx_luma(sample_input(uv).rgb); }
    return h * p.height_depth;
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let radius = max(p.radius, 1.0 / p.size_.x);
    let h0 = hgt(uv);
    let r = hash22(c, p.seed_ + 1.0);
    let fall = mix(4.0, 0.25, p.relief_balance);
    var occ = 0.0;
    for (var k = 0; k < 8; k++) {
        let a = (f32(k) + r.x) / 8.0 * 2.0 * PI;
        let dir = vec2f(cos(a), sin(a));
        var sin_h = 0.0;
        for (var i = 0; i < 10; i++) {
            let t = max((f32(i) + r.y) / 10.0 * radius, 1.0 / p.size_.x);
            let dh = hgt(uv + dir * t) - h0;
            let s = dh / sqrt(dh * dh + t * t);
            if (s > sin_h) {
                occ += (s - sin_h) * pow(1.0 - t / radius, fall);
                sin_h = s;
            }
        }
    }
    store_ao(c, vec4f(vec3f(clamp(1.0 - occ / 8.0 * p.intensity * 2.0, 0.0, 1.0)), 1.0));
}
