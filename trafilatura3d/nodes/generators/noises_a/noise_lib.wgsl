// Shared helpers of the noises_a generators (prepended to every shader of this folder).
// Only built-ins (p.size_, p.seed_) are used here: every node has them.

const TAU = 6.28318530717959;

// --- hashing ----------------------------------------------------------------------------------

// Hash of a lattice point wrapped on a period (the base of every tileable lattice).
fn hw(q: vec2i, per: vec2i, s: f32) -> f32 { return hash21(wrap(q, max(per, vec2i(1))), s); }
fn hw2(q: vec2i, per: vec2i, s: f32) -> vec2f { return hash22(wrap(q, max(per, vec2i(1))), s); }
fn hseed(i: i32, s: f32) -> f32 { return hash21(vec2i(i, 913), s); }

fn h3(q: vec3i, s: f32) -> f32 {
    return f32(pcg(u32(q.x) ^ pcg(u32(q.y) ^ pcg(u32(q.z) ^ pcg(bitcast<u32>(s)))))) / 4294967295.0;
}
fn h33(q: vec3i, s: f32) -> vec3f { return vec3f(h3(q, s), h3(q, s + 31.7), h3(q, s + 63.1)); }
// Wraps a 3D lattice index on `per` (axes with per <= 0 are not wrapped).
fn w3(q: vec3i, per: vec3i) -> vec3i {
    var r = q;
    if (per.x > 0) { r.x = wrap1(q.x, per.x); }
    if (per.y > 0) { r.y = wrap1(q.y, per.y); }
    if (per.z > 0) { r.z = wrap1(q.z, per.z); }
    return r;
}

// --- 2D domain ------------------------------------------------------------------------------

// Non-square expansion: keep cells square on non-square images (the long axis gets more cells).
fn aspect(nse: i32) -> vec2f {
    if (nse != 0) { return p.size_ / min(p.size_.x, p.size_.y); }
    return vec2f(1.0);
}
fn uv_of(c: vec2i) -> vec2f { return (vec2f(c) + 0.5) / p.size_; }
fn rot2(v: vec2f, turns: f32) -> vec2f {
    let a = turns * TAU;
    let cs = cos(a);
    let sn = sin(a);
    return vec2f(cs * v.x - sn * v.y, sn * v.x + cs * v.y);
}
// Direction of a Disorder displacement: random (h) over a span narrowed by the anisotropy.
fn disorder_dir(h: f32, aniso: f32, angle: f32) -> vec2f {
    let a = angle * TAU + (h - 0.5) * TAU * (1.0 - clamp(aniso, 0.0, 1.0));
    return vec2f(cos(a), sin(a));
}
// Pixel footprint in cells of a lattice with `cells` cells across the image width.
fn px_cells(cells: f32) -> f32 { return cells / p.size_.x; }

// --- 2D lattice noises (periodic) -------------------------------------------------------------

fn quintic(f: vec2f) -> vec2f { return f * f * f * (f * (f * 6.0 - 15.0) + 10.0); }

// Value noise in [0, 1].
fn vnoise(x: vec2f, per: vec2i, s: f32) -> f32 {
    let i = vec2i(floor(x));
    let u = quintic(fract(x));
    let a = hw(i, per, s);
    let b = hw(i + vec2i(1, 0), per, s);
    let c = hw(i + vec2i(0, 1), per, s);
    let d = hw(i + vec2i(1, 1), per, s);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn grad_at(q: vec2i, per: vec2i, s: f32) -> vec2f {
    let a = hw(q, per, s) * TAU;
    return vec2f(cos(a), sin(a));
}
// Gradient (Perlin) noise, roughly in [-1, 1].
fn gnoise(x: vec2f, per: vec2i, s: f32) -> f32 {
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = quintic(f);
    let a = dot(grad_at(i, per, s), f);
    let b = dot(grad_at(i + vec2i(1, 0), per, s), f - vec2f(1.0, 0.0));
    let c = dot(grad_at(i + vec2i(0, 1), per, s), f - vec2f(0.0, 1.0));
    let d = dot(grad_at(i + vec2i(1, 1), per, s), f - vec2f(1.0, 1.0));
    return 1.4 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// One octave by kind: 0 value, 1 gradient, 2 turbulence |g|, 3 ridge 1-|g|; result in [0, 1].
fn octave(x: vec2f, per: vec2i, s: f32, kind: i32) -> f32 {
    if (kind == 0) { return vnoise(x, per, s); }
    let g = clamp(gnoise(x, per, s), -1.0, 1.0);
    if (kind == 1) { return 0.5 + 0.5 * g; }
    if (kind == 2) { return abs(g); }
    return 1.0 - abs(g);
}

// Octaves from a base period until the finest one is ~2 pixels (more detail at higher resolution,
// same look), capped.
fn auto_octaves(per0: f32, cap: i32) -> i32 {
    return clamp(i32(floor(log2(max(p.size_.x, p.size_.y) / max(per0 * 2.0, 1.0)))) + 1, 1, cap);
}

// Fractal sum: x in [0,1]-tile units (already scaled by aspect), base period per0, octaves doubling,
// amplitude multiplied by `rough` per octave. Each octave is shifted by its own Disorder
// displacement (`dis` = disorder * speed, in cells of the octave). Result normalized to [0, 1].
fn fbm(x: vec2f, per0: vec2i, octaves: i32, rough: f32, s: f32, kind: i32,
       dis: f32, an: f32, ang: f32) -> f32 {
    var sum = 0.0;
    var wsum = 0.0;
    var w = 1.0;
    var per = max(per0, vec2i(1));
    for (var o = 0; o < octaves; o++) {
        let dir = disorder_dir(hseed(o, s), an, ang);
        sum += w * octave(x * vec2f(per) + dir * dis, per, s + f32(o) * 13.17, kind);
        wsum += w;
        w *= rough;
        per *= 2;
    }
    return sum / max(wsum, 1e-6);
}

// --- 2D cellular (periodic) -------------------------------------------------------------------

struct Cell2 {
    f1: f32,
    f2: f32,
    edge: f32,      // distance to the nearest cell border (Euclidean bisector distance)
    id: vec2i,      // wrapped id of the nearest site
    site: vec2f,    // nearest site, relative to x
};

// Sites: one per cell, jittered by `jit`, plus a Disorder offset moving each site on a short
// closed path (phase dis), oriented by the anisotropy. Distances in cells.
fn site_of(q: vec2i, per: vec2i, s: f32, jit: f32, dis: f32, an: f32, ang: f32) -> vec2f {
    let h = hw2(q, per, s);
    let k = hw2(q, per, s + 5.71);
    let dir = disorder_dir(k.x, an, ang);
    return vec2f(q) + 0.5 + (h - 0.5) * jit + dir * 0.3 * sin(TAU * (k.y + dis));
}

// metric: Minkowski order pk (2 Euclidean, 1 Manhattan, large -> Chebyshev).
fn mdist2(d: vec2f, pk: f32) -> f32 {
    if (abs(pk - 2.0) < 1e-3) { return length(d); }
    let a = abs(d);
    if (pk > 16.0) { return max(a.x, a.y); }
    return pow(pow(a.x, pk) + pow(a.y, pk), 1.0 / pk);
}

fn cells2(x: vec2f, per: vec2i, s: f32, jit: f32, dis: f32, an: f32, ang: f32, pk: f32) -> Cell2 {
    let i = vec2i(floor(x));
    var r: Cell2;
    r.f1 = 1e9;
    r.f2 = 1e9;
    r.edge = 1e9;
    var best = vec2i(0);
    for (var y = -2; y <= 2; y++) {
        for (var xx = -2; xx <= 2; xx++) {
            let q = i + vec2i(xx, y);
            let d = site_of(q, per, s, jit, dis, an, ang) - x;
            let dd = mdist2(d, pk);
            if (dd < r.f1) {
                r.f2 = r.f1;
                r.f1 = dd;
                r.site = d;
                best = q;
            } else if (dd < r.f2) {
                r.f2 = dd;
            }
        }
    }
    r.id = wrap(best, max(per, vec2i(1)));
    // Border distance: to the bisectors between the nearest site and its neighbours.
    for (var y = -2; y <= 2; y++) {
        for (var xx = -2; xx <= 2; xx++) {
            let q = best + vec2i(xx, y);
            if (xx == 0 && y == 0) { continue; }
            let d = site_of(q, per, s, jit, dis, an, ang) - x;
            let e = d - r.site;
            if (dot(e, e) > 1e-8) {
                r.edge = min(r.edge, dot(0.5 * (r.site + d), normalize(e)));
            }
        }
    }
    return r;
}

// --- 3D noises ----------------------------------------------------------------------------------

fn grad3(q: vec3i, per: vec3i, s: f32) -> vec3f {
    let h = h33(w3(q, per), s);
    let z = h.x * 2.0 - 1.0;
    let a = h.y * TAU;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3f(r * cos(a), r * sin(a), z);
}

// 3D Perlin noise, roughly in [-1, 1]; per <= 0 on an axis: not periodic.
fn perlin3(x: vec3f, per: vec3i, s: f32) -> f32 {
    let i = vec3i(floor(x));
    let f = fract(x);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var n: array<f32, 8>;
    for (var k = 0; k < 8; k++) {
        let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        n[k] = dot(grad3(i + o, per, s), f - vec3f(o));
    }
    let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
    let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
    return 2.0 * mix(x0, x1, u.z);
}

// Simplex-type noise on the body-centred cubic lattice (two interleaved cubic grids, radial
// kernels): isotropic like simplex, and periodic with an integer period like a cubic lattice.
fn simplex3(x: vec3f, per: vec3i, s: f32) -> f32 {
    var v = 0.0;
    for (var g = 0; g < 2; g++) {
        let y = x - vec3f(0.5 * f32(g));
        let i = vec3i(floor(y));
        let f = fract(y);
        for (var k = 0; k < 8; k++) {
            let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
            let d = f - vec3f(o);
            let t = 0.75 - dot(d, d);
            if (t > 0.0) {
                let t2 = t * t;
                v += t2 * t2 * dot(grad3(i + o, per, s + f32(g) * 101.0), d);
            }
        }
    }
    return clamp(v * 9.0, -1.0, 1.0);
}

fn mdist3(d: vec3f, mode: i32, pk: f32) -> f32 {
    let a = abs(d);
    if (mode == 1) { return a.x + a.y + a.z; }
    if (mode == 2) { return max(a.x, max(a.y, a.z)); }
    if (mode == 3) {
        let q = max(pk, 0.05);
        if (q > 30.0) { return max(a.x, max(a.y, a.z)); }
        let m = max(a.x, max(a.y, a.z)) + 1e-9;  // normalized to avoid overflow
        let b = a / m;
        return m * pow(pow(b.x, q) + pow(b.y, q) + pow(b.z, q), 1.0 / q);
    }
    return length(d);
}

struct Cell3 {
    f1: f32,
    f2: f32,
    id: vec3i,
};

// Cellular noise on one jittered point per cubic cell (jitter per axis in [0, 1]).
fn voronoi3(x: vec3f, per: vec3i, jit: vec3f, mode: i32, pk: f32, s: f32) -> Cell3 {
    let i = vec3i(floor(x));
    var r: Cell3;
    r.f1 = 1e9;
    r.f2 = 1e9;
    var best = vec3i(0);
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var xx = -1; xx <= 1; xx++) {
                let q = i + vec3i(xx, y, z);
                let site = vec3f(q) + 0.5 + (h33(w3(q, per), s) - 0.5) * clamp(jit, vec3f(0.0), vec3f(1.0));
                let dd = mdist3(site - x, mode, pk);
                if (dd < r.f1) {
                    r.f2 = r.f1;
                    r.f1 = dd;
                    best = q;
                } else if (dd < r.f2) {
                    r.f2 = dd;
                }
            }
        }
    }
    r.id = w3(best, per);
    return r;
}

// sRGB encoding of a linear value.
fn to_srgb(v: vec3f) -> vec3f {
    let c = clamp(v, vec3f(0.0), vec3f(1.0));
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

// --- 3D frame of the "3D" noises -------------------------------------------------------------

// Noise-space frequency per axis: Scale * Size, stepped to integers when tiling (the doc: "when
// Tiling is enabled the scale adjustment is stepped").
fn freq3(scale: f32, size: vec3f, tiled: bool) -> vec3f {
    let f = max(scale * size, vec3f(1e-3));
    if (tiled) { return max(round(f), vec3f(1.0)); }
    return f;
}
fn per3(f: vec3f, tiled: bool) -> vec3i {
    if (tiled) { return vec3i(round(f)); }
    return vec3i(0);
}
// Warp of the noise-space point x (u: unit-space position) by a Perlin vector field whose scale is
// the noise's times `mult`; intensity in cells of the noise.
fn distort3(x: vec3f, u: vec3f, f: vec3f, tiled: bool, intensity: f32, mult: f32, s: f32) -> vec3f {
    if (intensity == 0.0) { return x; }
    let fd = freq3(1.0, f * max(mult, 1e-3), tiled);
    let per = per3(fd, tiled);
    let q = u * fd;
    return x + intensity * vec3f(perlin3(q, per, s + 1.0), perlin3(q + vec3f(5.2, 1.3, 7.1), per, s + 2.0),
                                 perlin3(q + vec3f(2.9, 8.3, 3.7), per, s + 3.0));
}
// The contrast factor: 1 / (1 - Contrast), Contrast 0 leaves the noise as it is.
fn contrast3(c: f32) -> f32 { return 1.0 / (1.0 - min(c, 0.99999)); }

// Unit-space position of a pixel: the slice z = 0.5 of the unit cube, or the position map.
fn unit_slice(c: vec2i) -> vec3f { return vec3f(uv_of(c), 0.5); }

// Sparse convolution: `k` impulses per cell of a periodic grid, each a Gaussian of `sigma` cells
// with a weight from `wfn` (0: signed in [-1, 1], 1: positive in [0.3, 1]); returns the sum (or the
// max for positive weights when `use_max`). Impulses move on short closed paths with Disorder.
fn gauss_field(x: vec2f, per: vec2i, k: i32, sigma: f32, s: f32, wfn: i32, use_max: bool,
               dis: f32, an: f32, ang: f32) -> f32 {
    let i = vec2i(floor(x));
    var acc = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var ii = -2; ii <= 2; ii++) {
            let q = i + vec2i(ii, j);
            for (var m = 0; m < k; m++) {
                let sm = s + f32(m) * 17.0;
                let d = site_of(q, per, sm, 1.0, dis, an, ang) - x;
                var w = hw(q, per, sm + 3.0);
                if (wfn == 0) { w = w * 2.0 - 1.0; } else { w = 0.3 + 0.7 * w; }
                let g = w * exp(-dot(d, d) / (2.0 * sigma * sigma));
                if (use_max) { acc = max(acc, g); } else { acc += g; }
            }
        }
    }
    return acc;
}

// --- Fractal sum (FX-Map of bells) -----------------------------------------------------------------

// One level of the Fractal sum: a bell (Gaussian, sigma half a cell: no grid ripple) at the centre of
// every cell of a periodic grid, with a random signed intensity in [-0.35, 0.35]. Zero mean; a 1-cell
// grid is one bump.
// The bell is separable and the hash is staged, so the 4 x 4 block costs 8 exp, 4 row hashes and 16 cell hashes.
fn bell_level(x: vec2f, per: vec2i, s: f32) -> f32 {
    let i = vec2i(floor(x));
    let pw = max(per, vec2i(1));
    let hs = hash_seed(s);
    var cx: array<i32, 4>;
    var ex: array<f32, 4>;
    for (var k = 0; k < 4; k++) {
        let q = i.x + k - 1;
        cx[k] = wrap1(q, pw.x);
        let d = f32(q) + 0.5 - x.x;
        ex[k] = exp(-2.0 * d * d);
    }
    var acc = 0.0;
    for (var j = 0; j < 4; j++) {
        let q = i.y + j - 1;
        let hr = hash_row(wrap1(q, pw.y), hs);
        let d = f32(q) + 0.5 - x.y;
        var row = 0.0;
        for (var k = 0; k < 4; k++) { row += (hash_cell(cx[k], hr) - 0.5) * ex[k]; }
        acc += row * exp(-2.0 * d * d);
    }
    return acc * 0.7;
}

// Levels lo..hi of bells (level L: 2^L cells across the tile, times the aspect), level L+1 weighted
// `ratio` times level L, averaged by weight; Disorder shifts each level on its own. Signed, zero mean.
fn bell_sum(x: vec2f, asp: vec2f, lo: i32, hi: i32, ratio: f32, s: f32, dis: f32) -> f32 {
    var sum = 0.0;
    var wsum = 0.0;
    var w = 1.0;
    let a = clamp(lo, 0, 13);
    let b = clamp(hi, a, 13);
    for (var l = a; l <= b; l++) {
        let per = vec2i(round(exp2(f32(l)) * asp));
        let dir = disorder_dir(hseed(l, s), 0.0, 0.0);
        sum += w * bell_level(x * vec2f(per) + dir * dis, per, s + f32(l) * 13.17);
        wsum += w;
        w *= max(ratio, 0.0);
    }
    return sum / max(wsum, 1e-6);
}

// The Fractal sum of pixel c: bell levels lo..hi weighted by Roughness (the ratio between levels), the
// sum around mid-grey times the contrast factor k, then times the Global opacity (above 1 burns).
fn fractal_sum_at(c: vec2i, nse: i32, lo: i32, hi: i32, rough: f32, k: f32, opacity: f32, s: f32,
                  dis: f32) -> f32 {
    let asp = aspect(nse);
    let n = bell_sum(uv_of(c) * asp, asp, lo, hi, rough, s, dis);
    return clamp((n * k + 0.5) * opacity, 0.0, 1.0);
}

// --- standardized fractal sums (rough noises) ---------------------------------------------------

// `fbm` with zero mean and unit variance whatever the octave count: the octaves are independent, so
// the normalized sum has the octave's spread times sqrt(sum w^2) / sum w. Octave mean / std by kind
// (value, gradient, turbulence, ridge) measured on the lattice noises above.
fn fbm_z(x: vec2f, per0: vec2i, octaves: i32, rough: f32, s: f32, kind: i32,
         dis: f32, an: f32, ang: f32) -> f32 {
    var w = 1.0;
    var w1 = 0.0;
    var w2 = 0.0;
    for (var o = 0; o < octaves; o++) {
        w1 += w;
        w2 += w * w;
        w *= rough;
    }
    let k = clamp(kind, 0, 3);
    let mean = select(select(select(0.7527, 0.2473, k == 2), 0.5, k == 1), 0.4988, k == 0);
    let sd = select(select(0.1744, 0.1513, k == 1), 0.2269, k == 0);
    return (fbm(x, per0, octaves, rough, s, kind, dis, an, ang) - mean) * w1 / (sqrt(w2) * sd);
}

// Skews a standardized value and keeps it standardized (a shifted lognormal): a > 0 stretches the
// dark tail, a < 0 the bright one, a = 0 is the identity.
fn skew_z(u: f32, a: f32) -> f32 {
    if (abs(a) < 1e-4) { return u; }
    let e = exp(a * a);
    return (sqrt(e) - exp(-a * u)) * sign(a) / sqrt((e - 1.0) * e);
}
