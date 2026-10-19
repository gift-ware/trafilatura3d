// Directional Noise Fibrous (ours): octaves of gradient noise (2 x Scale x 2^o cells across the tile, equal weights),
// each averaged along the direction over a streak DF_SMEAR0 (coarsest) to DF_SMEAR (finest) of its cells long: a
// periodic lattice sampled along a line, so it tiles at any Angle. Angle random turns each octave's direction by a
// smooth noise of half its cells (up to that many turns either way), shortens the streaks and softens the finest
// octaves; octaves are faded out above DF_CUT cells across the tile (the grain stays the same at any Scale) and
// under two pixels; a higher Scale lowers the contrast (Scale^-DF_SCALE_FADE). Disorder moves each octave back and
// forth along its own direction (anisotropy narrows it), a full turn looping. All constants fitted on the output
// (mean, spread, octave bands, look).

const DF_OCTAVES = 9;
const DF_SMEAR0 = 1.5;
const DF_SMEAR = 3.0;
const DF_TAPS = 6;
const DF_MEAN = 0.425;
const DF_GAIN = 0.324;
const DF_CUT = 380.0;
const DF_ARCUT = 3.0;
const DF_ARSHORT = 0.8;
const DF_SCALE_FADE = 0.25;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let n = f32(max(p.scale, 1));
    let q = uv_of(c) * asp + p.tile_offset;
    let dis = p.disorder * p.disorder_speed;
    let px = 1.0 / max(p.size_.x / asp.x, p.size_.y / asp.y);
    let cut = DF_CUT / (1.0 + DF_ARCUT * p.angle_random);
    var sum = 0.0;
    var cells = 2.0 * n;
    for (var o = 0; o < DF_OCTAVES; o++) {
        if (cells * px * 2.0 > 1.0) { break; }  // under two pixels: not drawn
        let per = vec2i(round(cells * asp));
        let s = p.seed_ + 11.3 * f32(o);
        let turn = vnoise(q * cells * 0.5, vec2i(round(0.5 * cells * asp)), s + 3.7);
        let dir = rot2(vec2f(1.0, 0.0), p.angle + p.angle_random * (2.0 * turn - 1.0));
        let h = hseed(o, s);
        let shift = disorder_dir(h, p.disorder_anisotropy, p.disorder_anisotropy_angle) * 0.25 * sin(TAU * (dis + h));
        let base = q * cells + shift;
        let smear = mix(DF_SMEAR0, DF_SMEAR, f32(o) / f32(DF_OCTAVES - 1)) * (1.0 - DF_ARSHORT * p.angle_random);
        var acc = 0.0;
        for (var k = 0; k < DF_TAPS; k++) {
            acc += gnoise(base + dir * ((f32(k) + 0.5) / f32(DF_TAPS) - 0.5) * smear, per, s);
        }
        let r = cells / cut;
        // sqrt(smear): a shorter average keeps more of the octave's spread, the same weight at every length.
        sum += exp(-r * r) * sqrt(smear / DF_SMEAR) * acc / f32(DF_TAPS);
        cells *= 2.0;
    }
    let v = clamp(DF_MEAN + DF_GAIN * pow(n, -DF_SCALE_FADE) * sum, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
