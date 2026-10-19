fn hald(i: vec3i, n: i32, w: i32) -> vec3f {
    let k = i.r + i.g * n + i.b * n * n;
    return texel_lut(vec2i(k % w, k / w)).rgb;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    let w = size_lut().x;
    let level = i32(round(pow(f32(w), 1.0 / 3.0)));
    let n = level * level;
    var o = v.rgb;
    if (LUT_CONNECTED && n >= 2) {
        let x = clamp(v.rgb, vec3f(0.0), vec3f(1.0)) * f32(n - 1);
        let i = vec3i(min(floor(x), vec3f(f32(n - 2))));
        let f = x - vec3f(i);
        let c00 = mix(hald(i, n, w), hald(i + vec3i(1, 0, 0), n, w), f.r);
        let c10 = mix(hald(i + vec3i(0, 1, 0), n, w), hald(i + vec3i(1, 1, 0), n, w), f.r);
        let c01 = mix(hald(i + vec3i(0, 0, 1), n, w), hald(i + vec3i(1, 0, 1), n, w), f.r);
        let c11 = mix(hald(i + vec3i(0, 1, 1), n, w), hald(i + vec3i(1, 1, 1), n, w), f.r);
        o = mix(mix(c00, c10, f.g), mix(c01, c11, f.g), f.b);
    }
    if (p.lut_intensity_by_alpha != 0) { o = mix(v.rgb, o, v.a); }
    store(c, vec4f(o, v.a));
}
