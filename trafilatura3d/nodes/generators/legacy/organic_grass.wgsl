// Grass blades (ours): pixel noise u^p.gr_sparse (sparse bright dots) filtered by a kernel made of a small round
// core (p.gr_core pixels wide) and two short lines (blades in two directions, p.gr_turns.x and .y turns from the
// vertical, p.gr_length pixels each way, their taps fading as exp(-t / p.gr_fade)); weights p.gr_weights.xyz (core,
// first line, second line), the lines' sum moved towards their larger by p.gr_weights.w. The kernel reads around the
// pixel moved by a smooth random field (value noise over cells p.gr_jitter_px pixels wide, up to about p.gr_jitter
// pixels, rounded: crumpled blades). Minus soft blotches (value noise over p.gr_cells and twice as many cells, weight
// p.gr_blotch). The image is 0.5 + p.gr_lift + p.gr_gain x that, clamped. The grain is the pixel's, as the legacy
// node's; it tiles.
const MAXL = 16;

fn noise_at(q: vec2i, d: vec2i, s: u32) -> f32 {
    let w = vec2i(select(select(q.x, q.x - d.x, q.x >= d.x), q.x + d.x, q.x < 0),
                  select(select(q.y, q.y - d.y, q.y >= d.y), q.y + d.y, q.y < 0));
    return f32(pcg(pcg(u32(w.x) ^ s) ^ (u32(w.y) * 0x9E3779B9u)) >> 8u) * (1.0 / 16777216.0) - 0.5;
}

// The blades' grain: u^p.gr_sparse (u uniform; above 1: sparse bright dots), centred.
fn dots_at(q: vec2i, d: vec2i, s: u32) -> f32 {
    return pow(noise_at(q, d, s) + 0.5, p.gr_sparse) - 1.0 / (p.gr_sparse + 1.0);
}

fn value_noise(uv: vec2f, n: i32, s: u32) -> f32 {
    let x = uv * f32(n);
    let i = vec2i(floor(x));
    let f = smoothstep(vec2f(0.0), vec2f(1.0), fract(x));
    let m = vec2i(n);
    let a = noise_at(wrap(i, m), m, s);
    let b = noise_at(wrap(i + vec2i(1, 0), m), m, s);
    let c = noise_at(wrap(i + vec2i(0, 1), m), m, s);
    let e = noise_at(wrap(i + vec2i(1, 1), m), m, s);
    return mix(mix(a, b, f.x), mix(c, e, f.x), f.y);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = pcg(bitcast<u32>(p.gr_seed));
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let jn = max(i32(round(f32(d.x) / p.gr_jitter_px)), 1);
    let jitter = 2.0 * p.gr_jitter * vec2f(value_noise(uv, jn, s + 41u), value_noise(uv, jn, s + 53u));
    let q = c + vec2i(round(jitter));
    var core = 0.0;
    let k2 = -0.5 / (p.gr_core * p.gr_core);
    for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
            core += exp(f32(j * j + k * k) * k2) * dots_at(q + vec2i(k, j), d, s);
        }
    }
    let da = vec2f(sin(6.2831853 * p.gr_turns.x), cos(6.2831853 * p.gr_turns.x));
    let db = vec2f(sin(6.2831853 * p.gr_turns.y), cos(6.2831853 * p.gr_turns.y));
    var la = 0.0;
    var lb = 0.0;
    for (var t = 1; t <= min(p.gr_length, MAXL); t++) {
        let w = exp(-f32(t) / p.gr_fade);
        let oa = vec2i(round(f32(t) * da));
        let ob = vec2i(round(f32(t) * db));
        la += w * (dots_at(q + oa, d, s) + dots_at(q - oa, d, s));
        lb += w * (dots_at(q + ob, d, s) + dots_at(q - ob, d, s));
    }
    let n = max(p.gr_cells, 1);
    let blotch = value_noise(uv, n, s + 17u) + 0.5 * value_noise(uv, 2 * n, s + 29u);
    let a = p.gr_weights.y * la;
    let b = p.gr_weights.z * lb;
    let lines = mix(a + b, max(a, b), p.gr_weights.w);
    let v = 0.5 + p.gr_lift + p.gr_gain * (p.gr_weights.x * core + lines) - p.gr_blotch * blotch;
    let o = clamp(v, 0.0, 1.0);
    store(c, vec4f(o, o, o, 1.0));
}
