// Helpers of the Paint filters and generators (prepended to every shader of this folder).

fn contrasted(x: f32, k: f32) -> f32 { return clamp((x - 0.5) * (1.0 + k * 8.0) + 0.5, 0.0, 1.0); }
fn level_in(x: f32, level: f32) -> f32 { return clamp(x + (level - 0.5) * 2.0, 0.0, 1.0); }

fn hash3(q: vec3i, seed: f32) -> f32 {
    let h = pcg(u32(q.x) * 73856093u ^ u32(q.y) * 19349663u ^ u32(q.z) * 83492791u ^ pcg(u32(seed * 1000.0)));
    return f32(h & 0xffffffu) / 16777215.0;
}
fn vnoise(x: vec3f, seed: f32) -> f32 {
    let i = vec3i(floor(x)); let f = fract(x); let u = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(hash3(i, seed), hash3(i + vec3i(1, 0, 0), seed), u.x), mix(hash3(i + vec3i(0, 1, 0), seed), hash3(i + vec3i(1, 1, 0), seed), u.x), u.y);
    let b = mix(mix(hash3(i + vec3i(0, 0, 1), seed), hash3(i + vec3i(1, 0, 1), seed), u.x), mix(hash3(i + vec3i(0, 1, 1), seed), hash3(i + vec3i(1, 1, 1), seed), u.x), u.y);
    return mix(a, b, u.z);
}
fn fbm(x: vec3f, seed: f32) -> f32 {
    var s = 0.0; var a = 0.5; var y = x;
    for (var k = 0; k < 5; k++) { s += a * vnoise(y, seed + f32(k) * 17.0); y = y * 2.03; a *= 0.5; }
    return s / 0.96875;
}

fn srgb_to_linear(c: vec3f) -> vec3f { return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045)); }
fn linear_to_srgb(c: vec3f) -> vec3f { let x = max(c, vec3f(0.0)); return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308)); }
fn uv_of(c: vec2i) -> vec2f { return (vec2f(c) + 0.5) / p.size_; }
