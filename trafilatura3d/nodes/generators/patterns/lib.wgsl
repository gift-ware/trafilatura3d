// Shared helpers of the pattern nodes: frames, randomness (after nodes/shapes.wgsl: edge, ramp, the built-in
// shapes).

const TAU = 6.28318530718;

// Rotation by `a` turns (counter-clockwise in a y-up frame, clockwise on screen).
fn rot(a: f32) -> mat2x2f {
    let c = cos(a * TAU);
    let s = sin(a * TAU);
    return mat2x2f(c, s, -s, c);
}

fn wrapi(i: i32, n: i32) -> i32 { return wrap1(i, max(n, 1)); }

// Output pixel centre in uv. With Non Square Expansion the frame stays square: the longer axis
// spans more than 1 (patterns repeat instead of stretching; sizes are powers of two so it tiles).
fn frame(c: vec2i, nse: i32) -> vec2f {
    let uv = (vec2f(c) + 0.5) / p.size_;
    if (nse == 1) { return uv * p.size_ / min(p.size_.x, p.size_.y); }
    return uv;
}
// The extent of the frame (1, 1), or e.g. (2, 1) with Non Square Expansion on a 2:1 image.
fn frame_size(nse: i32) -> vec2f {
    if (nse == 1) { return p.size_ / min(p.size_.x, p.size_.y); }
    return vec2f(1.0);
}
// One output pixel in uv (the larger of the two axes).
fn pixel_uv() -> f32 { return 1.0 / min(p.size_.x, p.size_.y); }

// Random value in [0, 1) for an integer cell, a salt (one per use) and the node seed.
fn rnd(cell: vec2i, salt: u32) -> f32 {
    let h = pcg(u32(cell.x) * 1597334677u ^ pcg(u32(cell.y) * 3812015801u ^ pcg(salt ^ bitcast<u32>(p.seed_))));
    return f32(h >> 8u) / 16777216.0;
}
fn rnd1(i: i32, salt: u32) -> f32 { return rnd(vec2i(i, 0x5bd1e995), salt); }
// Signed random in [-1, 1).
fn srnd(cell: vec2i, salt: u32) -> f32 { return rnd(cell, salt) * 2.0 - 1.0; }
// A colour with its hue, saturation, lightness or alpha (`channel` 1-4; 0 = unchanged) multiplied by `f`.
fn hsla_scale(c: vec4f, channel: i32, f: f32) -> vec4f {
    if (channel == 0) { return c; }
    if (channel == 4) { return vec4f(c.rgb, c.a * f); }
    var h = rgb_to_hsl(clamp(c.rgb, vec3f(0.0), vec3f(1.0)));
    h[channel - 1] *= f;
    return vec4f(hsl_to_rgb(h), c.a);
}
// The Color Random of the colour tile nodes, per tile: hue shifted by up to +-r.x / 2, saturation and
// lightness moved towards random values by r.y and r.z, alpha towards a random fraction of itself by r.w.
fn color_random(c: vec4f, r: vec4f, cell: vec2i, salt: u32) -> vec4f {
    if (all(r == vec4f(0.0))) { return c; }
    var h = rgb_to_hsl(clamp(c.rgb, vec3f(0.0), vec3f(1.0)));
    h = vec3f(fract(h.x + r.x * (rnd(cell, salt) - 0.5)), mix(h.y, rnd(cell, salt + 1u), r.y),
              mix(h.z, rnd(cell, salt + 2u), r.z));
    return vec4f(hsl_to_rgb(h), mix(c.a, c.a * rnd(cell, salt + 3u), r.w));
}

// Approximately normal (sum of 3 uniforms), in [-1, 1].
fn grnd(cell: vec2i, salt: u32) -> f32 {
    return (rnd(cell, salt) + rnd(cell, salt + 101u) + rnd(cell, salt + 202u)) / 1.5 - 1.0;
}

