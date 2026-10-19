// Moisture noise (ours), shared by Moisture Noise, Moisture Noise (Legacy) and Clouds Soft (Legacy): a fractal sparse
// convolution noise around a gray. Octave o has 4 x Scale x 2^o cells across the tile (Pattern size divides them per
// axis, its x across the image's y, rounded: the noise tiles) and weight p.mo_rough^o (the first times p.mo_first), at
// most p.mo_octaves octaves and none finer than
// two pixels; on every cell a compact bell (1 - r^2)^2 of radius MO_RADIUS cells and a random height in -1..1 at a
// random point (Pattern angle + random turns it, Disorder moves it along a loop). The bells' compact support leaves
// flat patches between them. From the third octave on, p.mo_cluster grows the detail where the coarser octaves are
// strong (a hybrid multifractal). The sum's gain (p.mo_gain, rising with log2(Scale) by p.mo_gain_up / up2, and with
// the pattern's area to the power p.mo_area) and its lift over the 0.5 gray (p.mo_lift) are each node's own fit;
// Global opacity scales both (default 1.6 = as fitted). p.mo_seed varies the random stream.

const MO_RADIUS = 0.9;

// The 3 x 3 cells around x; the turn is the octave's, so its rotation and the hashes' seed stages are taken once.
fn mo_octave(x: vec2f, per: vec2i, s: f32, turn: f32, shift: vec2f) -> f32 {
    let i = vec2i(floor(x));
    let cs = cos(TAU * turn);
    let sn = sin(TAU * turn);
    let hx = hash_seed(s);
    let hy = hash_seed(s + 71.31);
    let hg = hash_seed(s + 5.1);
    var v = 0.0;
    for (var j = -1; j <= 1; j++) {
        let qy = i.y + j;
        let wy = wrap1(qy, per.y);
        let rx = hash_row(wy, hx);
        let ry = hash_row(wy, hy);
        let rg = hash_row(wy, hg);
        for (var k = -1; k <= 1; k++) {
            let qx = i.x + k;
            let wx = wrap1(qx, per.x);
            let h = vec2f(hash_cell(wx, rx), hash_cell(wx, ry)) - 0.5;
            let g = hash_cell(wx, rg);
            let at = vec2f(f32(qx), f32(qy)) + 0.5 + vec2f(cs * h.x - sn * h.y, sn * h.x + cs * h.y) * 0.5 + shift;
            let r2 = dot(x - at, x - at) / (MO_RADIUS * MO_RADIUS);
            let b = max(1.0 - r2, 0.0);
            v += (2.0 * g - 1.0) * b * b;
        }
    }
    return v;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion) + p.tile_offset;
    let px = max(p.size_.x, p.size_.y);
    let ps = max(p.pattern_size.yx, vec2f(1e-3));
    let seed = p.seed_ + p.mo_seed;
    var v = 0.0;
    var w = 1.0;
    var q = 4.0 * f32(max(p.scale, 1));
    for (var o = 0; o < min(p.mo_octaves, 12); o++) {
        if (q * 2.0 > px) { break; }
        let per = max(vec2i(round(q / ps)), vec2i(1));
        let h = hash22(vec2i(o, 4041), seed);
        let turn = p.pattern_angle + (h.x - 0.5) * p.pattern_angle_random;
        let shift = disorder_disp(h, p.disorder, p.disorder_speed, p.disorder_anisotropy, p.disorder_anisotropy_angle);
        let cluster = select(1.0, clamp(p.mo_cluster_base + p.mo_cluster * abs(v), 0.0, 2.5), o >= 2);
        v += select(w, w * p.mo_first, o == 0) * cluster * mo_octave(uv * vec2f(per), per, seed + f32(o) * 17.3, turn, shift);
        w *= p.mo_rough;
        q *= 2.0;
    }
    let ls = log2(f32(max(p.scale, 1)));
    let gain = p.mo_gain * (1.0 + p.mo_gain_up * clamp(ls, 0.0, 1.0) + p.mo_gain_up2 * clamp(ls - 1.0, 0.0, 1.0))
             * pow(ps.x * ps.y, p.mo_area);
    let k = p.global_opacity / 1.6;
    let g = sat(0.5 + k * (p.mo_lift + gain * v));
    store(c, vec4f(g, g, g, 1.0));
}
