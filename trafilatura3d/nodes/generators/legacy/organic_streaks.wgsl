// Fractal streaks (ours): a sparse convolution of stretched bells over octaves, the streaks along y (p.st_axis 1
// swaps the axes: along x). Octave 0 is a lattice of p.st_cells_x columns by p.st_cells_y rows over the tile; every
// octave doubles the columns, and the rows from octave p.st_delay on (the first octaves' cells grow longer relative
// to their width); the first p.st_bands octaves are bands, one bell per column stretched over the whole height.
// Every cell holds one bell (1 - r^2)^2 at a random point (p.st_jitter cells around the centre),
// half p.st_width cells wide and p.st_len_lo..p.st_len_hi cells long (uniform), its height u - p.st_center (u uniform:
// 0 = bright streaks on black, 0.5 = signed). Octave weights: p.st_weights for the first four, then the fourth's
// times p.st_rough per octave; octaves whose columns are narrower than p.st_min_px pixels are skipped; p.st_shift
// slides each octave along the streaks by a random part of the image (Disorder). The image is (p.st_lift + p.st_gain
// x sum) clamped, to the power p.st_gamma. The search covers 2 p.st_reach_x + 1 columns and 2 p.st_reach + 1 rows
// (the caller sizes them on the widest and longest bell).
const MAXR = 4;

fn unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
fn halves(h: u32) -> vec2f { return vec2f(f32(h & 0xFFFFu), f32(h >> 16u)) * (1.0 / 65536.0); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var uv = (vec2f(c) + 0.5) / vec2f(d);
    var px = vec2f(d);
    if (p.st_axis != 0) { uv = uv.yx; px = px.yx; }
    let ry = clamp(p.st_reach, 1, MAXR);
    let rx = clamp(p.st_reach_x, 1, 2);
    let inv_w = 1.0 / p.st_width;
    let seed = pcg(bitcast<u32>(p.st_seed));
    var n = vec2f(p.st_cells_x, p.st_cells_y);
    var w = p.st_weights.x;
    var sum = 0.0;
    var cols: array<u32, 5>;
    var rows: array<u32, 9>;
    for (var o = 0; o < p.st_octaves; o++) {
        if (px.x / n.x < p.st_min_px) { break; }
        let per = vec2i(n);
        let so = pcg(seed + u32(o) * 7919u);
        let x = (uv + vec2f(0.0, p.st_shift * unit(pcg(so)))) * n;
        let i = vec2i(floor(x));
        for (var k = -rx; k <= rx; k++) { cols[k + 2] = pcg(u32(wrap1(i.x + k, per.x)) ^ so); }
        let band = o < p.st_bands;  // a band: one bell per column, the full height of the tile
        let r = select(ry, 0, band);
        for (var j = -r; j <= r; j++) {
            rows[j + MAXR] = select(u32(wrap1(i.y + j, per.y)) * 0x9E3779B9u, 0u, band);
        }
        var acc = 0.0;
        for (var j = -r; j <= r; j++) {
            for (var k = -rx; k <= rx; k++) {
                let h0 = pcg(cols[k + 2] ^ rows[j + MAXR]);
                let lh = halves(pcg(h0));  // length, height
                let at = vec2f(i + vec2i(k, j)) + 0.5 + (halves(h0) - 0.5) * p.st_jitter;
                let e = (x - at) * vec2f(inv_w, select(1.0 / mix(p.st_len_lo, p.st_len_hi, lh.x), 0.0, band));
                let b = max(1.0 - dot(e, e), 0.0);
                acc += (lh.y - p.st_center) * b * b;
            }
        }
        sum += w * acc;
        w = select(w * p.st_rough, p.st_weights[min(o + 1, 3)], o < 3);
        n *= vec2f(2.0, select(1.0, 2.0, o + 1 > p.st_delay));
    }
    let v = pow(clamp(p.st_lift + p.st_gain * sum, 0.0, 1.0), p.st_gamma);
    store(c, vec4f(v, v, v, 1.0));
}
