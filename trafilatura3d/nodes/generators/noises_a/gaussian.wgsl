// Gaussian noise (ours, a sparse convolution noise): on every cell of the Scale grid one Gaussian bell, sigma
// GN_SIGMA cells, at a random point up to GN_JITTER cells from the cell's centre (Disorder turns that point around
// the centre, the anisotropy flattening the turn across its angle), summed and brought to the mean and spread
// fitted (GN_GAIN, GN_OFFSET). Gaussian spots 1 (ours): one jittered positive Gaussian bell
// (sigma 0.25 sub-cells) per sub-cell of a grid three times finer than Scale, summed, shifted down and clamped
// into spots on black.

const GN_SIGMA = 0.46;
const GN_JITTER = 1.2;
const GN_GAIN = 0.174;
const GN_OFFSET = 0.0873;

fn gn_point(q: vec2i, per: vec2i, dis: f32, aniso: f32, aniso_angle: f32) -> vec2f {
    let h = hw2(q, per, p.seed_ + 0.613);
    let rest = (h - 0.5) * 2.0 * GN_JITTER;
    let turned = rot2(rest, dis) - rest;
    let along = rot2(vec2f(1.0, 0.0), aniso_angle);
    let across = vec2f(-along.y, along.x);
    return rest + along * dot(turned, along) + across * dot(turned, across) * (1.0 - clamp(aniso, 0.0, 1.0));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per = vec2i(round(f32(max(p.scale, 1)) * asp));
    let x = (uv_of(c) * asp + p.tile_offset) * vec2f(per) / asp;
    let dis = p.disorder * p.disorder_speed;
    var v = 0.0;
    if (VARIANT == 0) {
        let i = vec2i(floor(x));
        for (var j = -3; j <= 3; j++) {
            for (var ii = -3; ii <= 3; ii++) {
                let q = i + vec2i(ii, j);
                let r = x - (vec2f(q) + 0.5 + gn_point(q, per, dis, p.disorder_anisotropy, p.disorder_anisotropy_angle));
                v += exp(-dot(r, r) / (2.0 * GN_SIGMA * GN_SIGMA));
            }
        }
        v = GN_OFFSET + GN_GAIN * v;
    } else {
        let sigma = 0.25;
        let g = gauss_field(x * 3.0, per * 3, 1, sigma, p.seed_, 1, false, dis, p.disorder_anisotropy,
                            p.disorder_anisotropy_angle);
        // Standardized: mean = density 1 x weight 0.65 x bell integral 2 pi sigma^2; spread measured.
        v = 0.12 + 0.14 * (g - 0.65 * TAU * sigma * sigma) / 0.238;
    }
    v = clamp(v, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
