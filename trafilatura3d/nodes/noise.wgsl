fn gradient(x: vec2f, period: i32, seed: f32) -> f32 {
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let per = vec2i(period);
    let g00 = dot(hash2(wrap(i, per), seed), f);
    let g10 = dot(hash2(wrap(i + vec2i(1, 0), per), seed), f - vec2f(1.0, 0.0));
    let g01 = dot(hash2(wrap(i + vec2i(0, 1), per), seed), f - vec2f(0.0, 1.0));
    let g11 = dot(hash2(wrap(i + vec2i(1, 1), per), seed), f - vec2f(1.0, 1.0));
    return mix(mix(g00, g10, u.x), mix(g01, g11, u.x), u.y);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    var v = 0.0;
    var amp = 0.5;
    var period = max(p.scale, 1);
    for (var o = 0; o < p.octaves; o++) {
        v += amp * gradient(uv * f32(period), period, p.seed_ + f32(o) * 17.0);
        amp *= 0.5;
        period *= 2;
    }
    let g = clamp(v * 0.75 + 0.5, 0.0, 1.0);
    store(c, vec4f(g, g, g, 1.0));
}
