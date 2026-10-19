// Stamps (ours): one layer of a sparse convolution noise, added to `under` (the coarser layers, drawn at their own size
// and read bilinearly). The tile has p.cells x p.cells lattice cells; every cell has p.count slots, each holding a stamp
// with probability p.density at a random point of the cell (Disorder moves it around the cell). The density is times
// a smooth random field when p.patchiness > 0 (the same field in every layer: dense and empty regions) and times a
// vertical profile when p.ramp = 1 (p.r0 .. p.r3 at the top, a third, two thirds and the bottom). A stamp is p.length
// cells long (at most 3; p.width_var shortens it at random) and p.aspect (at most 1) times that wide, turned at random
// when p.turn = 1; its profile p.shape (0 pyramid 1 - max(|x|, |y|), 1 flat disc with a one pixel edge, 2 bell
// (1 - r^2)^2, 3 paraboloid 1 - r^2) and its height uniform in p.h_low .. p.h_high. A stamp's centre is in its cell and
// it reaches length / 2 from it: the (floor(length) + 1)^2 cells around a pixel hold every stamp that reaches it.
// The layer's sum (clamped to 0-1 when p.clamp_sum = 1) times p.weight is added to `under`, plus p.grain x a white
// noise in -0.5..0.5 of p.grain_cells x p.grain_cells cells (one per pixel when 0); the last layer (p.last = 1) gives
// p.lift + that, clamped, through a gamma p.gamma.

fn st_u(h: u32) -> f32 { return f32(h) / 4294967295.0; }

fn st_lattice(q: vec2i, m: i32, s: u32) -> f32 {
    let w = wrap(q, vec2i(m));
    return st_u(pcg(u32(w.x) + u32(w.y) * 7919u + s));
}

fn st_field(uv: vec2f, m: i32, s: u32) -> f32 {
    // a smooth value noise over m x m cells (tiling)
    let f = uv * f32(m) - 0.5;
    let i = vec2i(floor(f));
    let t = smoothstep(vec2f(0.0), vec2f(1.0), fract(f));
    let a = mix(st_lattice(i, m, s), st_lattice(i + vec2i(1, 0), m, s), t.x);
    let b = mix(st_lattice(i + vec2i(0, 1), m, s), st_lattice(i + vec2i(1, 1), m, s), t.x);
    return mix(a, b, t.y);
}

fn st_profile(e: vec2f, px: f32) -> f32 {
    // e: the offset in half extents; px: a pixel in the same unit
    if (p.shape == 0) { return max(1.0 - max(abs(e.x), abs(e.y)), 0.0); }
    let r2 = dot(e, e);
    if (p.shape == 1) { return clamp((1.0 - sqrt(r2)) / px + 0.5, 0.0, 1.0); }
    let b = max(1.0 - r2, 0.0);
    return select(b * b, b, p.shape == 3);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let n = max(p.cells, 1);
    let sf = pcg(bitcast<u32>(p.seed_ + 0.37));  // the density field's seed: the same for every layer
    let sb = sf ^ pcg(u32(p.salt) + 1u);
    let len = clamp(p.length, 0.01, 3.0);
    let reach = i32(floor(len));
    let half = 0.5 * len;
    let x = uv * f32(n);
    let i = vec2i(floor(x - half));
    let px = f32(n) / (p.size_.x * half * p.aspect);
    var cols: array<i32, 4>;  // the wrapped columns, once (integer modulo is slow)
    for (var k = 0; k <= reach; k++) { cols[k] = wrap1(i.x + k, n); }
    var v = 0.0;
    for (var j = 0; j <= reach; j++) {
        let row = wrap1(i.y + j, n);
        let hr = pcg(u32(row) ^ sb);
        for (var k = 0; k <= reach; k++) {
            let col = cols[k];
            var dens = p.density;
            if (p.patchiness > 0.0) {
                let f = (vec2f(f32(col), f32(row)) + 0.5) / f32(n);
                let m = st_field(f, p.patch_cells, sf + 41u) + 0.5 * st_field(f, 2 * p.patch_cells, sf + 43u);
                dens *= clamp(1.0 + p.patchiness * (m / 1.5 - 0.5), 0.0, 4.0);
            }
            if (p.ramp != 0) {
                let y = clamp(3.0 * (f32(i.y + j) + 0.5) / f32(n), 0.0, 3.0);
                let lo = select(select(p.r2, p.r1, y < 2.0), p.r0, y < 1.0);
                let hi = select(select(p.r3, p.r2, y < 2.0), p.r1, y < 1.0);
                dens *= max(mix(lo, hi, y - min(floor(y), 2.0)), 0.0);
            }
            var h = pcg(u32(col) ^ hr);
            let base = vec2f(i + vec2i(k, j));
            for (var t = 0; t < p.count; t++) {
                h = pcg(h + 0x9e3779b9u);
                if (st_u(h) < dens) {
                    let h1 = pcg(h ^ 0x68bc21ebu);
                    let h2 = pcg(h1);
                    var at = vec2f(st_u(h1), st_u(h2));
                    if (p.disorder != 0.0) {
                        let a = 6.28318531 * st_u(h2 ^ 0x2545f491u);
                        at = fract(at + p.disorder * vec2f(cos(a), sin(a)));
                    }
                    var e = x - base - at;
                    if (dot(e, e) >= half * half) { continue; }
                    let h3 = pcg(h2);
                    if (p.turn != 0) {
                        let b = 6.28318531 * st_u(h3 ^ 0x5bd1e995u);
                        let cb = cos(b);
                        let sn = sin(b);
                        e = vec2f(cb * e.x + sn * e.y, cb * e.y - sn * e.x);
                    }
                    let g = p.h_low + (p.h_high - p.h_low) * st_u(h3);
                    var sc = 1.0 / half;
                    if (p.width_var != 0.0) { sc /= 1.0 - p.width_var * st_u(pcg(h3)); }
                    v += g * st_profile(e * vec2f(sc, sc / p.aspect), px * sc * half);
                }
            }
        }
    }
    if (p.clamp_sum != 0) { v = clamp(v, 0.0, 1.0); }
    var o = p.weight * v;
    if (UNDER_CONNECTED) { o += sample_under(uv).r; }
    if (p.grain != 0.0) {
        let q = select(vec2i(floor(uv * f32(p.grain_cells))), c, p.grain_cells <= 0);
        o += p.grain * (st_u(pcg(u32(q.x) ^ pcg(u32(q.y) ^ pcg(sb + 7u)))) - 0.5);
    }
    if (p.last != 0) { o = pow(clamp(p.lift + o, 0.0, 1.0), p.gamma); }
    store(c, vec4f(o, o, o, 1.0));
}
