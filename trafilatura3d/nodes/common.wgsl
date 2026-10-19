// Helpers available to every node shader.

// Texel coordinate under the node's tiling: wraps on tiled axes, clamps on the others. A texel inside is returned as
// it is: the integer remainders cost more than the reads around them (blur taps), and most reads are inside.
fn tile_coord(q: vec2i, d: vec2i) -> vec2i {
    if (all(q >= vec2i(0)) && all(q < d)) { return q; }
    var r = clamp(q, vec2i(0), d - 1);
    if ((p.tiling_ & 1) != 0) { r.x = wrap1(q.x, d.x); }
    if ((p.tiling_ & 2) != 0) { r.y = wrap1(q.y, d.y); }
    return r;
}

// k / 65535 correctly rounded, in integers: k / 65535 = n / (2^32 - 1) with n = k * 65537, a little above n / 2^32,
// so n rounded to 24 bits with the dropped bits as sticky (a half rounds up). Neither a division written in the shader
// (Metal) nor unpack2x16unorm (NVIDIA, Vulkan) is correctly rounded everywhere; reading an r16unorm texture is.
fn unorm16(k: vec4u) -> vec4f {
    let n = k * 65537u;
    let shift = vec4u(max(vec4i(0), 8 - vec4i(countLeadingZeros(n))));
    let dropped = shift > vec4u(0u);
    let half = vec4u(1u) << (max(shift, vec4u(1u)) - 1u);
    let up = select(vec4u(0u), vec4u(1u), dropped & ((n & ((vec4u(1u) << shift) - 1u)) >= half));
    return ldexp(vec4f((n >> shift) + up), vec4i(shift) - 32);
}

// A result quantized to an integer depth (q: 255 or 65535 levels - 1) as an integer texture holds it: k / q exactly,
// so a 16 bit result is the same value whether the device stores it as an integer or as a 32 bit float.
fn quantize_depth(v: vec4f, q: f32) -> vec4f {
    let k = vec4u(round(clamp(v, vec4f(0.0), vec4f(1.0)) * q));
    if (q > 255.0) { return unorm16(k); }
    return unpack4x8unorm(k.x | (k.y << 8u) | (k.z << 16u) | (k.w << 24u));
}

// The same quantization for a store into an integer texture, which rounds what it receives to k again: within an ulp
// of k / q is enough, and the unpack builtins are cheaper than unorm16 (4% of a 16 bit Levels at 4096).
fn quantize_for_unorm(v: vec4f, q: f32) -> vec4f {
    let k = vec4u(round(clamp(v, vec4f(0.0), vec4f(1.0)) * q));
    if (q > 255.0) { return vec4f(unpack2x16unorm(k.x | (k.y << 16u)), unpack2x16unorm(k.z | (k.w << 16u))); }
    return unpack4x8unorm(k.x | (k.y << 8u) | (k.z << 16u) | (k.w << 24u));
}

// Integer hash (PCG), the same on every GPU: the base of every random value.
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
// hash21 in three stages, for a loop over a block of cells: the seed's stage once, a row's once per row, then one
// stage per cell (the same values as hash21).
fn hash_seed(seed: f32) -> u32 { return pcg(bitcast<u32>(seed)); }
fn hash_row(y: i32, hs: u32) -> u32 { return pcg(u32(y) ^ hs); }
fn hash_cell(x: i32, hr: u32) -> f32 { return f32(pcg(u32(x) ^ hr)) / 4294967295.0; }
fn hash21(q: vec2i, seed: f32) -> f32 { return hash_cell(q.x, hash_row(q.y, hash_seed(seed))); }
fn hash22(q: vec2i, seed: f32) -> vec2f {
    return vec2f(hash21(q, seed), hash21(q, seed + 71.31));
}
fn hash2(q: vec2i, seed: f32) -> vec2f { return hash22(q, seed) * 2.0 - 1.0; }

fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

// HSL of an RGB colour: hue in turns (0..1), saturation, lightness.
fn rgb_to_hsl(c: vec3f) -> vec3f {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let l = (mx + mn) * 0.5;
    let dl = mx - mn;
    if (dl < 1e-6) { return vec3f(0.0, 0.0, l); }
    let s = dl / max(1.0 - abs(2.0 * l - 1.0), 1e-6);
    var h: f32;
    if (mx == c.r) { h = (c.g - c.b) / dl; }
    else if (mx == c.g) { h = (c.b - c.r) / dl + 2.0; }
    else { h = (c.r - c.g) / dl + 4.0; }
    return vec3f(fract(h / 6.0 + 1.0), s, l);
}

// RGB of an HSL colour; the hue wraps.
fn hsl_to_rgb(hsl: vec3f) -> vec3f {
    let k = (vec3f(0.0, 8.0, 4.0) + fract(hsl.x) * 12.0) % vec3f(12.0);
    let a = hsl.y * min(hsl.z, 1.0 - hsl.z);
    return hsl.z - a * clamp(min(k - 3.0, 9.0 - k), vec3f(-1.0), vec3f(1.0));
}

// The HSL node: `d` added to the hue (wrapping), saturation and lightness (clamped) of the colour as it is (a value
// below 0 from a float blend counts, its saturation past 1 too: measured on the original), the result clamped.
fn hsl_shift(c: vec3f, d: vec3f) -> vec3f {
    let h = rgb_to_hsl(c);
    return clamp(hsl_to_rgb(vec3f(h.x + d.x, clamp(h.yz + d.yz, vec2f(0.0), vec2f(1.0)))), vec3f(0.0), vec3f(1.0));
}

// The Contrast/Luminosity node: contrast about mid gray (1 = hard step, -1 = flat), then luminosity added.
fn contrast_luminosity(v: vec3f, con: f32, lum: f32) -> vec3f {
    let k = select(1.0 + con, 1.0 / max(1.0 - con, 1e-4), con > 0.0);
    return clamp((v - 0.5) * k + 0.5 + lum, vec3f(0.0), vec3f(1.0));
}

const PI = 3.14159265358979;
