// ---- materials/lib.wgsl: pure helpers shared by the material, mesh based and HDRI nodes ----

const TAU = 6.28318530717959;

fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(x: vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(1.0)); }
fn g4(v: f32) -> vec4f { return vec4f(v, v, v, 1.0); }
fn uv_of(c: vec2i, d: vec2i) -> vec2f { return (vec2f(c) + 0.5) / vec2f(d); }

fn srgb2lin(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(pow((x + 0.055) / 1.055, vec3f(2.4)), x / 12.92, x <= vec3f(0.04045));
}
fn lin2srgb(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// Histogram scan: as `pos` goes 0 -> 1 the input is revealed from its brightest values down to all
// white; `con` narrows the transition (1 = a hard threshold).
fn hscan(x: f32, pos: f32, con: f32) -> f32 {
    let w = max(1.0 - con, 0.002);
    let t = mix(1.0 + 0.5 * w, -0.5 * w, pos);
    return sat((x - t) / w + 0.5);
}
// Contrast about 0.5 (0 = unchanged, 1 = hard step) then a brightness shift (level 0.5 = none).
fn contrast_level(x: f32, con: f32, level: f32) -> f32 {
    return sat((x - 0.5) / max(1.0 - con, 0.002) + 0.5 + (level - 0.5) * 2.0);
}
fn smooth01(a: f32, b: f32, x: f32) -> f32 { return smoothstep(min(a, b - 1e-4), b, x); }

fn rgb2hsv(c: vec3f) -> vec3f {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let dl = mx - mn;
    var h = 0.0;
    if (dl > 1e-6) {
        if (mx == c.r) { h = (c.g - c.b) / dl; }
        else if (mx == c.g) { h = (c.b - c.r) / dl + 2.0; }
        else { h = (c.r - c.g) / dl + 4.0; }
        h = fract(h / 6.0 + 1.0);
    }
    return vec3f(h, select(0.0, dl / mx, mx > 1e-6), mx);
}
fn hsv2rgb(c: vec3f) -> vec3f {
    let k = fract(vec3f(c.x) + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0;
    return c.z * mix(vec3f(1.0), sat3(abs(k) - 1.0), c.y);
}
fn saturate_color(c: vec3f, s: f32) -> vec3f { return mix(vec3f(luma(c)), c, s); }
// The HSL node's hue and saturation (0.5 = unchanged; hue shifts the wheel, saturation 0..1 scales by 0..2).
fn hsl_adjust(c: vec3f, hue: f32, s: f32) -> vec3f {
    let h = rgb_to_hsl(sat3(c));
    return hsl_to_rgb(vec3f(h.x + hue - 0.5, sat(h.y * s * 2.0), h.z));
}

// --- noise (periodic: `n` cells across the unit square) -------------------------------------
fn cell_hash(q: vec2i, n: vec2i, seed: f32) -> f32 { return hash21(wrap(q, n), seed); }
fn vnoise(uv: vec2f, n: vec2i, seed: f32) -> f32 {
    let x = uv * vec2f(n);
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let a = cell_hash(i, n, seed);
    let b = cell_hash(i + vec2i(1, 0), n, seed);
    let c = cell_hash(i + vec2i(0, 1), n, seed);
    let d = cell_hash(i + vec2i(1, 1), n, seed);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn gnoise(uv: vec2f, n: vec2i, seed: f32) -> f32 {  // gradient noise, about 0..1
    let x = uv * vec2f(n);
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var r = 0.0;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var k = 0; k < 4; k++) {
        let o = vec2i(k & 1, k >> 1);
        let a = cell_hash(i + o, n, seed) * TAU;
        acc[k] = dot(vec2f(cos(a), sin(a)), f - vec2f(o));
    }
    r = mix(mix(acc[0], acc[1], u.x), mix(acc[2], acc[3], u.x), u.y);
    return r * 0.9 + 0.5;
}
// Fractal sum of `oct` octaves of gradient noise from `base` cells, each octave x2 cells.
fn fbm(uv: vec2f, base: i32, oct: i32, rough: f32, seed: f32) -> f32 {
    var s = 0.0;
    var w = 1.0;
    var tot = 0.0;
    var n = max(base, 1);
    for (var o = 0; o < oct; o++) {
        s += w * gnoise(uv, vec2i(n), seed + f32(o) * 13.7);
        tot += w;
        w *= rough;
        n *= 2;
    }
    return s / tot;
}
// Worley cells: x = F1, y = F2 (in cell units), z = random value of the nearest cell, w = edge (F2-F1).
fn cells(uv: vec2f, n: i32, jitter: f32, seed: f32) -> vec4f {
    let x = uv * f32(n);
    let i = vec2i(floor(x));
    let f = fract(x);
    var f1 = 9.0;
    var f2 = 9.0;
    var id = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
            let o = vec2i(k, j);
            let q = wrap(i + o, vec2i(n));
            let pt = vec2f(o) + 0.5 + (hash22(q, seed) - 0.5) * jitter;
            let dd = length(pt - f);
            if (dd < f1) { f2 = f1; f1 = dd; id = hash21(q, seed + 5.0); }
            else if (dd < f2) { f2 = dd; }
        }
    }
    return vec4f(f1, f2, id, f2 - f1);
}
// Grunge: fbm with cell-based spots, the generic "dirt" breakup of the mask generators.
fn grunge(uv: vec2f, scale: f32, seed: f32) -> f32 {
    let n = i32(max(scale, 1.0));
    let a = fbm(uv, n * 2, 5, 0.55, seed);
    let s = cells(uv, n * 4, 1.0, seed + 3.0);
    let spots = smoothstep(0.55, 0.15, s.x) * s.z;
    return sat(a * 0.75 + spots * 0.45 - 0.1);
}
// Scratches: one random segment per cell (length in cells, `ang` = main direction, `spread` random).
fn scratches(uv: vec2f, n: i32, len: f32, ang: f32, spread: f32, width: f32, seed: f32) -> f32 {
    let x = uv * f32(n);
    let i = vec2i(floor(x));
    var v = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var k = -2; k <= 2; k++) {
            let o = vec2i(k, j);
            let q = wrap(i + o, vec2i(n));
            let h = hash22(q, seed);
            let a = ang + (hash21(q, seed + 9.0) - 0.5) * spread * PI;
            let dir = vec2f(cos(a), sin(a));
            let ctr = vec2f(i + o) + h;
            let rel = x - ctr;
            let t = clamp(dot(rel, dir), -0.5 * len, 0.5 * len);
            let dd = length(rel - dir * t);
            let fade = 1.0 - abs(t) / max(0.5 * len, 1e-3);
            v = max(v, smoothstep(width, 0.0, dd) * (0.4 + 0.6 * hash21(q, seed + 2.0)) * (0.5 + 0.5 * fade));
        }
    }
    return v;
}

// --- normals --------------------------------------------------------------------------------
fn n_dec(c: vec3f, directx: bool) -> vec3f {
    var n = c * 2.0 - 1.0;
    if (directx) { n.y = -n.y; }
    return normalize(n + vec3f(0.0, 0.0, 1e-5));
}
fn n_enc(n: vec3f, directx: bool) -> vec3f {
    var m = normalize(n);
    if (directx) { m.y = -m.y; }
    return m * 0.5 + 0.5;
}
// Detail normal on top of a base normal (whiteout blend).
fn n_combine(base: vec3f, det: vec3f) -> vec3f {
    return normalize(vec3f(base.xy + det.xy, base.z * det.z));
}
// Tangent normal from height slopes (heights per pixel), `k` = intensity.
fn n_slopes(dx: f32, dy: f32, k: f32) -> vec3f { return normalize(vec3f(-dx * k, -dy * k, 1.0)); }
// Tangent normal from four height samples one pixel apart (left, right, up, down), intensity k.
fn n_from4(l: f32, r: f32, u: f32, dn: f32, k: f32) -> vec3f { return n_slopes((r - l) * 0.5, (u - dn) * 0.5, k); }
fn n_lerp(a: vec3f, b: vec3f, t: f32) -> vec3f { return normalize(mix(a, b, t) + vec3f(0.0, 0.0, 1e-4)); }

// --- blending ---------------------------------------------------------------------------------
// Material blending modes: Normal, Add, Subtract, Multiply, Add/Sub, Max, Min, Switch.
fn blend_mode(m: i32, b: vec3f, f: vec3f) -> vec3f {
    switch m {
        case 1: { return b + f; }
        case 2: { return max(b - f, vec3f(0.0)); }
        case 3: { return b * f; }
        case 4: { return b + 2.0 * f - 1.0; }
        case 5: { return max(b, f); }
        case 6: { return min(b, f); }
        default: { return f; }
    }
}
// Mask from a Color ID map: 1 at `tgt`, fading to 0 at distance `fuzz`; `pad` hardens the edge.
fn id_mask(c: vec3f, tgt: vec3f, fuzz: f32, pad: f32) -> f32 {
    let d = distance(c, tgt) / 1.7320508;
    let r = max(fuzz, 1e-3);
    return 1.0 - smoothstep(r * pad * 0.999, r, d);
}

// --- geometry -------------------------------------------------------------------------------
fn rot2(v: vec2f, a: f32) -> vec2f { let c = cos(a); let s = sin(a); return vec2f(c * v.x - s * v.y, s * v.x + c * v.y); }
fn inv2(m: vec4f) -> mat2x2f {  // inverse of the Transformation's 2x2 matrix (columns m.xy, m.zw)
    let det = m.x * m.w - m.y * m.z;
    if (abs(det) < 1e-8) { return mat2x2f(0.0, 0.0, 0.0, 0.0); }
    return mat2x2f(m.w, -m.y, -m.z, m.x) * (1.0 / det);
}
fn rot_axis(axis: vec3f, a: f32) -> mat3x3f {
    let u = normalize(axis);
    let c = cos(a);
    let s = sin(a);
    let t = 1.0 - c;
    return mat3x3f(
        vec3f(t * u.x * u.x + c, t * u.x * u.y + s * u.z, t * u.x * u.z - s * u.y),
        vec3f(t * u.x * u.y - s * u.z, t * u.y * u.y + c, t * u.y * u.z + s * u.x),
        vec3f(t * u.x * u.z + s * u.y, t * u.y * u.z - s * u.x, t * u.z * u.z + c));
}
// Orthonormal frame around a direction: (tangent, bitangent, n).
fn frame(n: vec3f) -> mat3x3f {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t = normalize(cross(up, n));
    return mat3x3f(t, cross(n, t), n);
}

// --- latitude / longitude panoramas (2:1, Y up, centre of the image looks along -Z) ----------
fn ll_dir(uv: vec2f) -> vec3f {
    let lon = (uv.x - 0.5) * TAU;
    let lat = (0.5 - uv.y) * PI;
    return vec3f(cos(lat) * sin(lon), sin(lat), -cos(lat) * cos(lon));
}
fn dir_ll(d: vec3f) -> vec2f {
    let n = normalize(d);
    return vec2f(atan2(n.x, -n.z) / TAU + 0.5, 0.5 - asin(clamp(n.y, -1.0, 1.0)) / PI);
}
// Ray from the origin along `d` against the horizontal plane y = h: distance, or -1.
fn ray_plane_y(d: vec3f, h: f32) -> f32 {
    if (abs(d.y) < 1e-5) { return -1.0; }
    let t = h / d.y;
    return select(-1.0, t, t > 0.0);
}

// Blackbody color (linear RGB, brightest channel 1) of a temperature in Kelvin.
fn blackbody(k: f32) -> vec3f {
    let t = clamp(k, 500.0, 40000.0) / 100.0;
    var r = 255.0;
    var g = 0.0;
    var b = 255.0;
    if (t > 66.0) {
        r = 329.698727446 * pow(t - 60.0, -0.1332047592);
        g = 288.1221695283 * pow(t - 60.0, -0.0755148492);
    } else {
        g = 99.4708025861 * log(t) - 161.1195681661;
        if (t <= 19.0) { b = 0.0; } else { b = 138.5177312231 * log(t - 10.0) - 305.0447927307; }
    }
    let c = srgb2lin(clamp(vec3f(r, g, b) / 255.0, vec3f(0.0), vec3f(1.0)));
    return c / max(c.r, max(c.g, max(c.b, 1e-4)));
}

fn tonemap_aces(x: vec3f) -> vec3f {
    return sat3((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}
fn tonemap_hejl(x: vec3f) -> vec3f {  // Hejl-Dawson filmic, output already display-encoded
    let y = max(vec3f(0.0), x - 0.004);
    return (y * (6.2 * y + 0.5)) / (y * (6.2 * y + 1.7) + 0.06);
}

// Triplanar weights from a world normal (contrast 0 = soft, 1 = hard).
fn tri_weights(n: vec3f, contrast: f32) -> vec3f {
    let k = mix(1.0, 32.0, sat(contrast));
    let w = pow(abs(n) + 1e-4, vec3f(k));
    return w / (w.x + w.y + w.z);
}
// Grunge projected triplanar from a position (0..1) and world normal.
fn grunge_tri(pos: vec3f, n: vec3f, contrast: f32, scale: f32, seed: f32) -> f32 {
    let w = tri_weights(n, contrast);
    return w.x * grunge(pos.zy, scale, seed) + w.y * grunge(pos.xz, scale, seed + 1.0) + w.z * grunge(pos.xy, scale, seed + 2.0);
}
