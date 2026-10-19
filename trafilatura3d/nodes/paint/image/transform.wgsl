// Input texel under the Repeat mode: wrapped (0), clamped (1); None (2) reads transparent black outside.
fn fetch_texel(q: vec2i) -> vec4f {
    let s = size_input();
    if (p.repeat_mode == 0) { return texel_input(wrap(q, s)); }
    if (p.repeat_mode == 2 && (any(q < vec2i(0)) || any(q >= s))) { return vec4f(0.0); }
    return texel_input(clamp(q, vec2i(0), s - 1));
}

fn fetch(uv: vec2f) -> vec4f {
    var q = uv;
    if (p.flip_mode == 1 || p.flip_mode == 3) { q.x = 1.0 - q.x; }
    if (p.flip_mode == 2 || p.flip_mode == 3) { q.y = 1.0 - q.y; }
    let x = q * vec2f(size_input()) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    return mix(mix(fetch_texel(i), fetch_texel(i + vec2i(1, 0)), f.x),
               mix(fetch_texel(i + vec2i(0, 1)), fetch_texel(i + vec2i(1, 1)), f.x), f.y);
}

fn effect(c: vec2i) -> vec4f {
    // Inverse of: input -> scale -> rotate -> offset, about the centre.
    let s = max(abs(p.scaling), 1e-3);
    let a = -p.rotation * 2.0 * PI;
    var ab = vec2f(cos(a), sin(a)) / s;
    if (p.tile_safe_rotation != 0) {
        ab = round(ab);
        if (all(ab == vec2f(0.0))) { ab = vec2f(1.0, 0.0); }
    }
    let v = uv_of(c) - 0.5 - p.offset - vec2f(p.offset_x, p.offset_y);
    let axes = vec2f(max(abs(p.scaling_x), 1e-3), max(abs(p.scaling_y), 1e-3));
    let q = vec2f(ab.x * v.x - ab.y * v.y, ab.y * v.x + ab.x * v.y) / axes + 0.5;
    // Footprint of one output pixel in input texels: average it when shrinking.
    let foot = length(ab) * max(1.0 / axes.x, 1.0 / axes.y) * max(f32(size_input().x) / p.size_.x, f32(size_input().y) / p.size_.y);
    let taps = clamp(i32(ceil(foot)), 1, 4);
    let du = vec2f(ab.x, ab.y) / axes / p.size_;
    let dv = vec2f(-ab.y, ab.x) / axes / p.size_;
    var acc = vec4f(0.0);
    for (var j = 0; j < taps; j++) {
        for (var i = 0; i < taps; i++) {
            let o = (vec2f(f32(i), f32(j)) + 0.5) / f32(taps) - 0.5;
            acc += fetch(q + du * o.x + dv * o.y);
        }
    }
    return acc / f32(taps * taps);
}
