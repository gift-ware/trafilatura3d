// Helpers of the MatFX and Stylization filters (nodes/paint/matfx/), prepended after paint/lib.wgsl.
// Noises here are periodic over the unit square (`n` cells across), so the filters' patterns tile.

fn mx_sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn mx_sat3(x: vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(1.0)); }
fn mx_uv(c: vec2i) -> vec2f { return (vec2f(c) + 0.5) / p.size_; }
fn mx_px() -> f32 { return p.size_.x / 256.0; }  // pixels per "pixel at 256 px"
fn mx_luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn mx_rot(v: vec2f, a: f32) -> vec2f { let c = cos(a); let s = sin(a); return vec2f(c * v.x - s * v.y, s * v.x + c * v.y); }
// Contrast around 0.5 (k in -1..1: -1 flat, 0 unchanged, 1 nearly a step).
fn mx_contrast(x: f32, k: f32) -> f32 {
    let g = select(1.0 + k, 1.0 / max(1.0 - k, 0.02), k > 0.0);
    return clamp((x - 0.5) * g + 0.5, 0.0, 1.0);
}
fn mx_contrast3(x: vec3f, k: f32) -> vec3f {
    let g = select(1.0 + k, 1.0 / max(1.0 - k, 0.02), k > 0.0);
    return clamp((x - 0.5) * g + 0.5, vec3f(0.0), vec3f(1.0));
}
// Histogram scan: `pos` moves the threshold (0 nothing, 1 everything), `con` hardens it (0 soft, 1 step).
fn mx_scan(x: f32, pos: f32, con: f32) -> f32 {
    let w = max(0.5 * (1.0 - con), 1e-3);
    let t = 1.0 - pos;
    return smoothstep(t - w, t + w, x * (1.0 - 2.0 * w) + w);
}

// --- noise --------------------------------------------------------------------------------------
fn mx_hash(q: vec2i, n: vec2i, seed: f32) -> f32 { return hash21(wrap(q, n), seed); }
fn mx_noise(uv: vec2f, n: vec2i, seed: f32) -> f32 {
    let x = uv * vec2f(n);
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mx_hash(i, n, seed), mx_hash(i + vec2i(1, 0), n, seed), u.x),
               mix(mx_hash(i + vec2i(0, 1), n, seed), mx_hash(i + vec2i(1, 1), n, seed), u.x), u.y);
}
// Fractal sum of `oct` octaves from `base` cells (x2 per octave), 0..1.
fn mx_fbm(uv: vec2f, base: i32, oct: i32, rough: f32, seed: f32) -> f32 {
    var s = 0.0; var w = 1.0; var tot = 0.0; var n = max(base, 1);
    for (var o = 0; o < oct; o++) {
        s += w * mx_noise(uv, vec2i(n), seed + f32(o) * 13.7);
        tot += w; w *= rough; n *= 2;
    }
    return s / tot;
}
// Worley cells: x = F1, y = F2 (cell units), z = random value of the nearest cell, w = F2 - F1.
fn mx_cells(uv: vec2f, n: vec2i, jitter: f32, seed: f32) -> vec4f {
    let x = uv * vec2f(n);
    let i = vec2i(floor(x));
    let f = fract(x);
    var f1 = 9.0; var f2 = 9.0; var id = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
            let o = vec2i(k, j);
            let q = wrap(i + o, n);
            let pt = vec2f(o) + 0.5 + (hash22(q, seed) - 0.5) * jitter;
            let dd = length(pt - f);
            if (dd < f1) { f2 = f1; f1 = dd; id = hash21(q, seed + 5.0); } else if (dd < f2) { f2 = dd; }
        }
    }
    return vec4f(f1, f2, id, f2 - f1);
}

// Scratches: a random segment per cell (`len` cells long, `width` cells wide), 1 on the lines.
fn mx_scratches(uv: vec2f, n: i32, len: f32, width: f32, seed: f32) -> f32 {
    let x = uv * f32(n);
    let i = vec2i(floor(x));
    var v = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
            let q = wrap(i + vec2i(k, j), vec2i(n));
            let r = hash22(q, seed);
            let ctr = vec2f(i + vec2i(k, j)) + r;
            let a = hash21(q, seed + 3.0) * PI;
            let d = vec2f(cos(a), sin(a));
            let l = len * (0.4 + 0.6 * hash21(q, seed + 7.0));
            let rel = x - ctr;
            let t = clamp(dot(rel, d), -l * 0.5, l * 0.5);
            let dist = length(rel - d * t);
            let taper = 1.0 - abs(t) / (l * 0.5 + 1e-4);
            v = max(v, smoothstep(width, 0.0, dist) * taper * step(0.4, hash21(q, seed + 9.0)));
        }
    }
    return v;
}

// --- colors -------------------------------------------------------------------------------------
fn mx_rgb2hsv(c: vec3f) -> vec3f {
    let mx = max(c.r, max(c.g, c.b)); let mn = min(c.r, min(c.g, c.b)); let d = mx - mn;
    var h = 0.0;
    if (d > 1e-6) {
        if (mx == c.r) { h = (c.g - c.b) / d + select(0.0, 6.0, c.g < c.b); }
        else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
        else { h = (c.r - c.g) / d + 4.0; }
        h /= 6.0;
    }
    return vec3f(h, select(0.0, d / mx, mx > 1e-6), mx);
}
fn mx_hsv2rgb(c: vec3f) -> vec3f {
    let k = abs(fract(c.x + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3f(1.0), clamp(k - 1.0, vec3f(0.0), vec3f(1.0)), c.y);
}
// Hue shift (turns), saturation and luminosity (0.5 neutral) of an sRGB color.
fn mx_hsl_adjust(c: vec3f, hue: f32, sat: f32, lum: f32) -> vec3f {
    var h = mx_rgb2hsv(mx_sat3(c));
    h.x = fract(h.x + hue);
    h.y = mx_sat(h.y * sat * 2.0);
    var r = mx_hsv2rgb(h);
    let l = (lum - 0.5) * 2.0;
    r = select(r * (1.0 + l), r + (1.0 - r) * l, l > 0.0);
    return mx_sat3(r);
}

// --- normals (tangent space, OpenGL: green = up = towards the previous row) ---------------------
fn mx_ndec(c: vec3f) -> vec3f { return normalize(c * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5)); }
fn mx_nenc(n: vec3f) -> vec3f { return normalize(n) * 0.5 + 0.5; }
// Detail normal over a base normal (whiteout).
fn mx_ndetail(base: vec3f, det: vec3f) -> vec3f { return normalize(vec3f(base.xy + det.xy, base.z * det.z)); }
// Normal of a relief from its slopes per pixel (dx towards +x, dy towards the previous row), intensity k.
fn mx_nslope(dx: f32, dy: f32, k: f32) -> vec3f { return normalize(vec3f(-dx * k, -dy * k, 1.0)); }

// Triplanar weights of a world space normal, harder with `hard` (0..1).
fn mx_tri(n: vec3f, hard: f32) -> vec3f {
    var w = pow(abs(n), vec3f(1.0 + hard * 15.0));
    return w / max(w.x + w.y + w.z, 1e-6);
}
