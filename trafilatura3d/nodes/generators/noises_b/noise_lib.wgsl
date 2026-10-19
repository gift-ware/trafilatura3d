// Shared building blocks of the noises_b generators. Everything works in UV (0-1 = one tile) on
// lattices with an integer period, wrapped, so every result tiles and is resolution independent.
// Randomness comes only from hash21/pcg seeded with the node seed.

const TAU = 6.28318530717959;

fn pix_uv(c: vec2i) -> vec2f { return (vec2f(c) + 0.5) / p.size_; }

// Non-square expansion: on non-square images keep the tile square and extend the pattern.
fn expand_uv(uv: vec2f, on: i32) -> vec2f {
    if (on == 0) { return uv; }
    return uv * p.size_ / min(p.size_.x, p.size_.y);
}

fn rot2(v: vec2f, a: f32) -> vec2f {
    let c = cos(a);
    let s = sin(a);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

fn hcell(q: vec2i, per: vec2i, s: f32) -> f32 { return hash21(wrap(q, per), s); }
fn hcell2(q: vec2i, per: vec2i, s: f32) -> vec2f { return hash22(wrap(q, per), s); }
fn hash31(q: vec3i, s: f32) -> f32 {
    return f32(pcg(u32(q.x) ^ pcg(u32(q.y) ^ pcg(u32(q.z) ^ pcg(bitcast<u32>(s)))))) / 4294967295.0;
}

fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn remap(x: f32, a: f32, b: f32) -> f32 { return sat((x - a) / max(b - a, 1e-5)); }

// Displacement of one noise "ingredient" by Disorder: a loop of radius 0.25 cell travelled by
// disorder * speed turns; anisotropy flattens the loop along the anisotropy angle (turns).
fn disorder_disp(h: vec2f, disorder: f32, speed: f32, aniso: f32, angle: f32) -> vec2f {
    let ph = TAU * (disorder * speed + h.x);
    let a = sat(aniso);
    let local = vec2f(cos(ph), (1.0 - a) * sin(ph));
    return 0.25 * rot2(local, TAU * angle + (1.0 - a) * TAU * h.y);
}

// ---- lattice noises -------------------------------------------------------------------------

// Value noise in [0, 1]; x in lattice units, period `per` cells.
fn vnoise(x: vec2f, per: vec2i, s: f32) -> f32 {
    let i = vec2i(floor(x));
    let f = fade2(fract(x));
    let a = hcell(i, per, s);
    let b = hcell(i + vec2i(1, 0), per, s);
    let c = hcell(i + vec2i(0, 1), per, s);
    let d = hcell(i + vec2i(1, 1), per, s);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Gradient (Perlin) noise in about [-1, 1]; each lattice gradient turns by `phase` turns (disorder).
fn gnoise_ph(x: vec2f, per: vec2i, s: f32, phase: f32) -> f32 {
    let i = vec2i(floor(x));
    let f = fract(x);
    let u = fade2(f);
    var r = array<f32, 4>();
    for (var k = 0; k < 4; k++) {
        let o = vec2i(k & 1, k >> 1);
        let h = hcell2(i + o, per, s);
        let dir = select(-1.0, 1.0, h.y > 0.5);
        let a = TAU * (h.x + phase * dir);
        r[k] = dot(vec2f(cos(a), sin(a)), f - vec2f(o));
    }
    return 1.41 * mix(mix(r[0], r[1], u.x), mix(r[2], r[3], u.x), u.y);
}
fn gnoise(x: vec2f, per: vec2i, s: f32) -> f32 { return gnoise_ph(x, per, s, 0.0); }

// Fractal sums over uv with base period `per` (doubling per octave), normalized.
fn fbm(uv: vec2f, per: i32, oct: i32, gain: f32, s: f32) -> f32 {  // value noise, [0, 1]
    var v = 0.0;
    var a = 1.0;
    var t = 0.0;
    var q = max(per, 1);
    for (var o = 0; o < oct; o++) {
        v += a * vnoise(uv * f32(q), vec2i(q), s + f32(o) * 13.7);
        t += a;
        a *= gain;
        q *= 2;
    }
    return v / t;
}
fn gfbm(uv: vec2f, per: i32, oct: i32, gain: f32, s: f32) -> f32 {  // gradient noise, ~[-1, 1]
    var v = 0.0;
    var a = 1.0;
    var t = 0.0;
    var q = max(per, 1);
    for (var o = 0; o < oct; o++) {
        v += a * gnoise(uv * f32(q), vec2i(q), s + f32(o) * 13.7);
        t += a;
        a *= gain;
        q *= 2;
    }
    return v / t;
}
// Contrasted fbm in [0, 1] (value noise clusters around 0.5).
fn cfbm(uv: vec2f, per: i32, oct: i32, gain: f32, s: f32) -> f32 {
    return sat(gfbm(uv, per, oct, gain, s) * 0.9 + 0.5);
}
// Ridged fbm: 1 at the zero crossings of gradient noise (long thin ridges).
fn ridged(uv: vec2f, per: i32, oct: i32, gain: f32, s: f32) -> f32 {
    var v = 0.0;
    var a = 1.0;
    var t = 0.0;
    var q = max(per, 1);
    for (var o = 0; o < oct; o++) {
        v += a * (1.0 - abs(gnoise(uv * f32(q), vec2i(q), s + f32(o) * 13.7)));
        t += a;
        a *= gain;
        q *= 2;
    }
    return v / t;
}
// Domain warp vector (two decorrelated gradient fbms).
fn warp_vec(uv: vec2f, per: i32, oct: i32, s: f32) -> vec2f {
    return vec2f(gfbm(uv, per, oct, 0.5, s), gfbm(uv, per, oct, 0.5, s + 41.3));
}
// Fine grain at a fixed density (resolution independent): value noise of period `per`.
fn grain(uv: vec2f, per: i32, s: f32) -> f32 {
    return vnoise(uv * f32(per), vec2i(per), s);
}

// ---- cells ----------------------------------------------------------------------------------

struct Cell {
    f1: f32,
    f2: f32,
    id: vec2i,      // wrapped cell of the nearest point
    rel: vec2f,     // x - nearest point (lattice units)
};

// Euclidean Voronoi on a jittered lattice, period `per` cells, x in lattice units.
fn voronoi(x: vec2f, per: vec2i, jitter: f32, s: f32) -> Cell {
    let i = vec2i(floor(x));
    var r = Cell(9.0, 9.0, vec2i(0), vec2f(0.0));
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            let pt = vec2f(q) + 0.5 + (hcell2(q, per, s) - 0.5) * jitter;
            let d = length(x - pt);
            if (d < r.f1) {
                r.f2 = r.f1;
                r = Cell(d, r.f2, wrap(q, per), x - pt);
            } else if (d < r.f2) {
                r.f2 = d;
            }
        }
    }
    return r;
}

// Distance to the cell border (exact, two-pass), for crisp crack lines.
fn voronoi_edge(x: vec2f, per: vec2i, jitter: f32, s: f32) -> f32 {
    let i = vec2i(floor(x));
    var best = 9.0;
    var mp = vec2f(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            let pt = vec2f(q) + 0.5 + (hcell2(q, per, s) - 0.5) * jitter;
            let d = dot(x - pt, x - pt);
            if (d < best) { best = d; mp = pt; }
        }
    }
    var e = 9.0;
    for (var y = -2; y <= 2; y++) {
        for (var xx = -2; xx <= 2; xx++) {
            let q = i + vec2i(xx, y);
            let pt = vec2f(q) + 0.5 + (hcell2(q, per, s) - 0.5) * jitter;
            let dd = pt - mp;
            if (dot(dd, dd) > 1e-6) {
                e = min(e, dot((mp + pt) * 0.5 - x, normalize(dd)));
            }
        }
    }
    return max(e, 0.0);
}

// ---- scattered ingredients ------------------------------------------------------------------

// Signed Gaussian blobs (`n` per cell, sigma in cells, window 0.75 cell), period `per`, displaced
// by Disorder. Sum around 0, roughly [-1, 1].
fn blobs(uv: vec2f, per: i32, n: i32, sigma: f32, s: f32, dis: vec4f) -> f32 {
    let x = uv * f32(per);
    let i = vec2i(floor(x));
    let pv = vec2i(per);
    var v = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            for (var k = 0; k < n; k++) {
                let sk = s + f32(k) * 7.31;
                let h = hcell2(q, pv, sk);
                let h2 = hcell2(q, pv, sk + 3.7);
                let pt = vec2f(q) + 0.25 + 0.5 * h + disorder_disp(h2, dis.x, dis.y, dis.z, dis.w);
                let d = x - pt;
                let r2 = dot(d, d);
                let w = max(1.0 - r2 / 0.5625, 0.0);
                let amp = hcell(q, pv, sk + 9.1) * 2.0 - 1.0;
                v += amp * exp(-r2 / (sigma * sigma)) * w * w;
            }
        }
    }
    return v;
}

// Discs: max over `n` discs per cell, radius in [rmin, rmax] cells (rmax <= 1), softness 0-1
// (1 = cone), probability `dens` that a disc exists. Returns vec2(coverage, value of the disc).
fn discs(uv: vec2f, per: i32, n: i32, dens: f32, rmin: f32, rmax: f32, soft: f32, s: f32) -> vec2f {
    let x = uv * f32(per);
    let i = vec2i(floor(x));
    let pv = vec2i(per);
    var best = vec2f(0.0);
    for (var y = -2; y <= 2; y++) {
        for (var xx = -2; xx <= 2; xx++) {
            let q = i + vec2i(xx, y);
            for (var k = 0; k < n; k++) {
                let sk = s + f32(k) * 5.13;
                if (hcell(q, pv, sk + 2.2) > dens) { continue; }
                let pt = vec2f(q) + hcell2(q, pv, sk);
                let hr = hcell2(q, pv, sk + 8.8);
                let r = mix(rmin, rmax, hr.x * hr.x);
                let d = length(x - pt) / r;
                let m = 1.0 - smoothstep(1.0 - max(soft, 0.02), 1.0, d);
                if (m > best.x) { best = vec2f(m, hr.y); }
            }
        }
    }
    return best;
}

// Distance from x to segment ab and the parameter along it.
fn seg(x: vec2f, a: vec2f, b: vec2f) -> vec2f {
    let ab = b - a;
    let t = clamp(dot(x - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    return vec2f(length(x - a - ab * t), t);
}

// Straight scratches: `n` segments per cell, direction angle +- angle_rand/2 (turns), length up
// to `len` cells (<= 1.8), half-width `w` cells, probability `dens`. Max of tapered lines, each
// with its own brightness in [vmin, 1].
fn scratches(uv: vec2f, per: i32, n: i32, dens: f32, angle: f32, angle_rand: f32, len: f32, w: f32,
             vmin: f32, s: f32) -> f32 {
    let x = uv * f32(per);
    let i = vec2i(floor(x));
    let pv = vec2i(per);
    var v = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            for (var k = 0; k < n; k++) {
                let sk = s + f32(k) * 3.71;
                let h = hcell2(q, pv, sk + 1.3);
                if (h.x > dens) { continue; }
                let c = vec2f(q) + hcell2(q, pv, sk);
                let h3 = hcell2(q, pv, sk + 6.1);
                let a = TAU * (angle + (h.y - 0.5) * angle_rand);
                let half = 0.5 * len * (0.35 + 0.65 * h3.x);
                let dir = vec2f(cos(a), sin(a)) * half;
                let r = seg(x, c - dir, c + dir);
                let taper = 1.0 - pow(abs(r.y * 2.0 - 1.0), 3.0);
                let ww = w * (0.4 + 0.6 * h3.y) * taper;
                let m = 1.0 - smoothstep(ww * 0.4, ww + 1e-4, r.x);
                v = max(v, m * mix(vmin, 1.0, hcell(q, pv, sk + 4.4)));
            }
        }
    }
    return v;
}

// Drips running down (+y): columns of `per.x` cells, rows of `per.y`; a cell starts a drip with
// probability `dens` at a random point, `len` rows long at most (<= 3), half-width `w` cells,
// slanted by up to `slant` cells per cell (<= 0.5), edge softness 1 - `crisp`. Max of mask x fade.
fn drips(uv: vec2f, per: vec2i, dens: f32, len: f32, w: f32, slant: f32, crisp: f32, s: f32) -> f32 {
    let x = uv * vec2f(per);
    let i = vec2i(floor(x));
    var v = 0.0;
    let soft = mix(0.95, 0.05, sat(crisp));
    for (var dy = -3; dy <= 1; dy++) {  // +1: the rounded top of a drip starting just below
        for (var dx = -2; dx <= 2; dx++) {
            let q = i + vec2i(dx, dy);
            if (hcell(q, per, s + 0.7) > dens) { continue; }
            let h = hcell2(q, per, s);
            let h2 = hcell2(q, per, s + 5.5);
            let top = vec2f(q) + h;
            let l = len * (0.3 + 0.7 * h2.x);
            let sl = slant * (h2.y * 2.0 - 1.0);
            let bottom = top + vec2f(sl * l, l);
            let wob = 0.05 * sin(TAU * ((x.y - top.y) * 0.9 + h.x * 5.0));
            let r = seg(x - vec2f(wob, 0.0), top, bottom);
            let wl = w * (0.6 + 0.8 * hcell(q, per, s + 9.0));
            let width = wl * (0.55 + 0.45 * r.y + 0.7 * smoothstep(0.8, 1.0, r.y));
            let m = 1.0 - smoothstep(width * (1.0 - soft), width + 1e-4, r.x);
            v = max(v, m * (1.0 - 0.45 * r.y));
        }
    }
    return v;
}

// Short curved strokes (shavings, curls): `n` arcs per cell, radius 0.12-0.4 cell, span 0.2-0.6
// turn, half-width `w` cells, probability `dens`. Max of tapered arcs with random brightness.
fn arcs(uv: vec2f, per: i32, n: i32, dens: f32, w: f32, s: f32) -> f32 {
    let x = uv * f32(per);
    let i = vec2i(floor(x));
    let pv = vec2i(per);
    var v = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            for (var k = 0; k < n; k++) {
                let sk = s + f32(k) * 6.37;
                let h = hcell2(q, pv, sk);
                if (hcell(q, pv, sk + 2.9) > dens) { continue; }
                let h2 = hcell2(q, pv, sk + 4.4);
                let c = vec2f(q) + h;
                let r = mix(0.12, 0.4, h2.x);
                let d = x - c;
                let span = mix(0.2, 0.6, h2.y);
                let ang = fract(atan2(d.y, d.x) / TAU - hcell(q, pv, sk + 8.1));
                if (ang > span) { continue; }
                let t = ang / span;
                let ww = w * (1.0 - pow(abs(2.0 * t - 1.0), 2.0));
                let m = 1.0 - smoothstep(ww * 0.3, ww + 1e-4, abs(length(d) - r));
                v = max(v, m * (0.5 + 0.5 * hcell(q, pv, sk + 3.3)));
            }
        }
    }
    return v;
}

// Stain: inside where the warped noise `n` is under `thr`, with a darker rim of width `rim`.
fn stain(n: f32, thr: f32, rim: f32) -> vec2f {
    let inside = 1.0 - smoothstep(thr - 0.01, thr + 0.01, n);
    let edge = 1.0 - smoothstep(0.0, rim, abs(n - thr));
    return vec2f(inside, edge);
}

// Galvanized spangles: Voronoi crystals, each a flat random gray crossed by dendritic streaks in
// its own direction. Returns vec2(crystal value, cell border distance).
fn spangles(uv: vec2f, per: i32, s: f32) -> vec2f {
    let c = voronoi(uv * f32(per), vec2i(per), 0.9, s);
    let base = hash21(c.id, s + 3.0);
    let a = TAU * hash21(c.id, s + 4.0);
    let l = rot2(c.rel, a);
    let streak = vnoise(vec2f(l.x * 1.5, l.y * 14.0) + vec2f(hash22(c.id, s + 5.0)) * 50.0, vec2i(1 << 20), s + 6.0);
    return vec2f(sat(base * 0.75 + 0.25 * streak), c.f2 - c.f1);
}

// ---- grunge finishing (Balance / Contrast / Invert) --------------------------------------------

fn balance_contrast(v: f32, balance: f32, contrast: f32) -> f32 {
    let b = v + (balance - 0.5);
    let k = 1.0 / max(1.0 - 0.98 * sat(contrast), 0.02);
    return sat((b - 0.5) * k + 0.5);
}
