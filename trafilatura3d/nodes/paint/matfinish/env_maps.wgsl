// Baked Lighting Environment's lighting maps, from a lat-long environment (pass_ 0) or its reduction (1, 2):
// 0: the environment averaged down to 64x32; 1: diffuse irradiance (cosine-weighted, / PI), 32x16;
// 2: the reflection prefiltered for roughness 0.4, 0.6, 0.8, 1.0, four 64x32 levels stacked vertically.
fn texel_angle(q: vec2i, d: vec2i) -> f32 {  // solid angle of a texel of a d-sized lat-long
    let lat = (0.5 - (f32(q.y) + 0.5) / f32(d.y)) * PI;
    return cos(lat) * (2.0 * PI / f32(d.x)) * (PI / f32(d.y));
}
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let src = size_environment();
    if (p.pass_ == 0) {
        let t = clamp(i32(ceil(f32(src.x) / p.size_.x)), 1, 16);
        var s = vec3f(0.0);
        for (var j = 0; j < t; j++) {
            for (var i = 0; i < t; i++) {
                s += sample_environment((vec2f(c) + (vec2f(f32(i), f32(j)) + 0.5) / f32(t)) / p.size_).rgb;
            }
        }
        store(c, vec4f(s / f32(t * t), 1.0));
        return;
    }
    var d = bl_ll_dir((vec2f(c) + 0.5) / p.size_);
    var e = 1.0;
    if (p.pass_ == 2) {
        let lv = c.y / 32;
        d = bl_ll_dir((vec2f(vec2i(c.x, c.y - lv * 32)) + 0.5) / vec2f(64.0, 32.0));
        let r = 0.4 + 0.2 * f32(lv);
        let a = r * r;
        e = 2.0 / (a * a) - 2.0;
    }
    var s = vec3f(0.0); var wsum = 0.0;
    for (var y = 0; y < src.y; y++) {
        for (var x = 0; x < src.x; x++) {
            let q = vec2i(x, y);
            let w = pow(max(dot(d, bl_ll_dir((vec2f(q) + 0.5) / vec2f(src))), 0.0), e) * texel_angle(q, src);
            s += texel_environment(q).rgb * w;
            wsum += w;
        }
    }
    // Irradiance / PI = the cosine-weighted mean (weights sum to PI); the reflection: the normalized lobe.
    store(c, vec4f(s / max(wsum, 1e-8), 1.0));
}
