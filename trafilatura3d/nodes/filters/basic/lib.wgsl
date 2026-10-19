// Helpers shared by the basic filters (prepended to each of their shaders).

fn uv_at(c: vec2i) -> vec2f { return (vec2f(c) + 0.5) / vec2f(size()); }

// Pixels of a 256 px image -> output pixels.
fn px256() -> f32 { return p.size_.x / 256.0; }

// True when uv falls off the image on an axis that does not tile (show background there).
fn off_image(uv: vec2f) -> bool {
    let ox = uv.x < 0.0 || uv.x > 1.0;
    let oy = uv.y < 0.0 || uv.y > 1.0;
    return (ox && !tile_h()) || (oy && !tile_v());
}

fn gray4(v: f32) -> vec4f { return vec4f(v, v, v, 1.0); }

// Rotation of a vector by `turns` (counter-clockwise in maths axes = clockwise on screen, y down).
fn rot(v: vec2f, turns: f32) -> vec2f {
    let a = turns * 2.0 * PI;
    let s = sin(a);
    let k = cos(a);
    return vec2f(k * v.x - s * v.y, s * v.x + k * v.y);
}

// --- colour ------------------------------------------------------------------------------------

fn srgb_to_linear(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(pow((x + 0.055) / 1.055, vec3f(2.4)), x / 12.92, x <= vec3f(0.04045));
}

fn linear_to_srgb(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

// Rec. 709 luma weights (the "official" luminance of linear sRGB primaries).
fn luma709(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// sRGB-encoded colour -> CIE Lab (D65), L in 0..100.
fn rgb_to_lab(c: vec3f) -> vec3f {
    let l = srgb_to_linear(c);
    let xyz = vec3f(dot(l, vec3f(0.4124, 0.3576, 0.1805)), dot(l, vec3f(0.2126, 0.7152, 0.0722)),
                    dot(l, vec3f(0.0193, 0.1192, 0.9505))) / vec3f(0.95047, 1.0, 1.08883);
    let f = select(pow(max(xyz, vec3f(0.0)), vec3f(1.0 / 3.0)), xyz * 7.787 + 16.0 / 116.0, xyz <= vec3f(0.008856));
    return vec3f(116.0 * f.y - 16.0, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z));
}

fn lab_to_rgb(lab: vec3f) -> vec3f {
    let fy = (lab.x + 16.0) / 116.0;
    let f = vec3f(fy + lab.y / 500.0, fy, fy - lab.z / 200.0);
    let xyz = select(f * f * f, (f - 16.0 / 116.0) / 7.787, f <= vec3f(0.2069)) * vec3f(0.95047, 1.0, 1.08883);
    let l = vec3f(dot(xyz, vec3f(3.2406, -1.5372, -0.4986)), dot(xyz, vec3f(-0.9689, 1.8758, 0.0415)),
                  dot(xyz, vec3f(0.0557, -0.2040, 1.0570)));
    return linear_to_srgb(l);
}

// CIEDE2000 colour difference between two Lab colours, with weights (kL, kC, kH).
fn delta_e2000(a: vec3f, b: vec3f, k: vec3f) -> f32 {
    let c1 = length(a.yz);
    let c2 = length(b.yz);
    let cm = 0.5 * (c1 + c2);
    let cm7 = pow(cm, 7.0);
    let g = 0.5 * (1.0 - sqrt(cm7 / (cm7 + 6103515625.0)));
    let a1 = a.y * (1.0 + g);
    let a2 = b.y * (1.0 + g);
    let cp1 = length(vec2f(a1, a.z));
    let cp2 = length(vec2f(a2, b.z));
    var h1 = atan2(a.z, a1);
    if (h1 < 0.0) { h1 += 2.0 * PI; }
    var h2 = atan2(b.z, a2);
    if (h2 < 0.0) { h2 += 2.0 * PI; }
    let dl = b.x - a.x;
    let dc = cp2 - cp1;
    var dh = h2 - h1;
    if (cp1 * cp2 == 0.0) { dh = 0.0; }
    else if (dh > PI) { dh -= 2.0 * PI; }
    else if (dh < -PI) { dh += 2.0 * PI; }
    let dhh = 2.0 * sqrt(cp1 * cp2) * sin(dh * 0.5);
    let lm = 0.5 * (a.x + b.x);
    let cpm = 0.5 * (cp1 + cp2);
    var hm = h1 + h2;
    if (cp1 * cp2 != 0.0) {
        if (abs(h1 - h2) > PI) { hm = select(hm - 2.0 * PI, hm + 2.0 * PI, hm < 2.0 * PI); }
        hm *= 0.5;
    }
    let t = 1.0 - 0.17 * cos(hm - PI / 6.0) + 0.24 * cos(2.0 * hm) + 0.32 * cos(3.0 * hm + PI / 30.0)
        - 0.20 * cos(4.0 * hm - 63.0 * PI / 180.0);
    let dtheta = 30.0 * PI / 180.0 * exp(-pow((hm * 180.0 / PI - 275.0) / 25.0, 2.0));
    let cpm7 = pow(cpm, 7.0);
    let rc = 2.0 * sqrt(cpm7 / (cpm7 + 6103515625.0));
    let l50 = (lm - 50.0) * (lm - 50.0);
    let sl = 1.0 + 0.015 * l50 / sqrt(20.0 + l50);
    let sc = 1.0 + 0.045 * cpm;
    let sh = 1.0 + 0.015 * cpm * t;
    let rt = -sin(2.0 * dtheta) * rc;
    let x = dl / (max(k.x, 1e-4) * sl);
    let y = dc / (max(k.y, 1e-4) * sc);
    let z = dhh / (max(k.z, 1e-4) * sh);
    return sqrt(max(x * x + y * y + z * z + rt * y * z, 0.0));
}

// Non-separable blend helpers (W3C compositing: luminosity / color modes).
fn lum_w3c(c: vec3f) -> f32 { return dot(c, vec3f(0.3, 0.59, 0.11)); }
fn clip_color(c: vec3f) -> vec3f {
    let l = lum_w3c(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    var o = c;
    if (n < 0.0) { o = l + (o - l) * l / max(l - n, 1e-6); }
    if (x > 1.0) { o = l + (o - l) * (1.0 - l) / max(x - l, 1e-6); }
    return o;
}
fn set_lum(c: vec3f, l: f32) -> vec3f { return clip_color(c + (l - lum_w3c(c))); }

// --- noise -------------------------------------------------------------------------------------

// Value noise with a lattice of `freq` cells over the unit square, periodic (tiles).
fn vnoise(uv: vec2f, freq: vec2i, seed: f32) -> f32 {
    let x = uv * vec2f(freq);
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(wrap(i, freq), seed);
    let b = hash21(wrap(i + vec2i(1, 0), freq), seed);
    let c = hash21(wrap(i + vec2i(0, 1), freq), seed);
    let d = hash21(wrap(i + vec2i(1, 1), freq), seed);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// A few octaves of value noise, 0..1, periodic over the unit square.
fn fbm(uv: vec2f, freq: vec2i, seed: f32) -> f32 {
    var s = 0.0;
    var amp = 0.5;
    var f = freq;
    for (var o = 0; o < 4; o++) {
        s += amp * vnoise(uv, f, seed + f32(o) * 13.7);
        amp *= 0.5;
        f *= 2;
    }
    return s / 0.9375;
}

// Quantize (Levels to (N - 1) / 255 at 8 bits and back): n evenly spaced levels including 0 and 1.
fn quantize_steps(v: vec4f, n: vec4f) -> vec4f {
    let k = max(n - 1.0, vec4f(1.0));
    return round(clamp(v, vec4f(0.0), vec4f(1.0)) * k) / k;
}

// Histogram Scan (one Levels): the input window centred on 1 - position, as wide as
// the part of 0..1 that fits around it (position 0: all black, 1: all white, 0.5: the full range), shrunk
// towards its centre by contrast (1: a hard step).
fn histogram_scan(v: f32, position: f32, contrast: f32) -> f32 {
    let q = 1.0 - position;
    let a = clamp(2.0 * q, 0.0, 1.0);
    let b = clamp(2.0 * q - 1.0, 0.0, 1.0);
    let k = clamp(contrast * 0.5, 0.0, 0.5);
    let lo = mix(b, a, k);
    let hi = mix(a, b, k);
    // The window is never narrower than 1e-4 (measured on the original: at Position 1 an input of 1e-5 gives 0.1, not
    // white); Position 0 stays black even at 1.
    return clamp((v - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
}

// Output UV -> "square" units (the short side spans 1), so rotations keep their shape.
fn square_scale() -> vec2f { return p.size_ / min(p.size_.x, p.size_.y); }

// Background colour as the output's type.
fn background(v: vec4f, gray: bool) -> vec4f { return select(v, gray4(v.r), gray); }

// Symmetry transfer modes: Copy, Add, Subtract, Multiply, Add Sub, Max, Min (a: the pixel, b: its
// reflection; alpha takes the larger of the two).
fn transfer(mode: i32, a: vec4f, b: vec4f) -> vec4f {
    var o: vec3f;
    switch mode {
        case 1: { o = a.rgb + b.rgb; }
        case 2: { o = max(a.rgb - b.rgb, vec3f(0.0)); }
        case 3: { o = a.rgb * b.rgb; }
        case 4: { o = a.rgb + 2.0 * b.rgb - 1.0; }
        case 5: { o = max(a.rgb, b.rgb); }
        case 6: { o = min(a.rgb, b.rgb); }
        default: { o = b.rgb; }
    }
    return vec4f(o, max(a.a, b.a));
}

// Where a pixel reads from under the Transformation matrices of Symmetry and Symmetry Slice (mirrors and quarter
// turns about the centre, UV y down): 0 left-right, 1 top-bottom, 2 main diagonal (TL-BR), 3 anti-diagonal
// (TR-BL), 4 point (half turn), 5 quarter turn reading (y, 1 - x), 6 the other one (1 - y, x).
fn reflect_uv(kind: i32, q: vec2f) -> vec2f {
    switch kind {
        case 0: { return vec2f(1.0 - q.x, q.y); }
        case 1: { return vec2f(q.x, 1.0 - q.y); }
        case 2: { return q.yx; }
        case 3: { return vec2f(1.0 - q.y, 1.0 - q.x); }
        case 5: { return vec2f(q.y, 1.0 - q.x); }
        case 6: { return vec2f(1.0 - q.y, q.x); }
        default: { return 1.0 - q; }
    }
}
