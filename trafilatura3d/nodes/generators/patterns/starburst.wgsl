// Error function (Abramowitz-Stegun 7.1.26, absolute error < 1.5e-7).
fn erf_as(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        * exp(-x * x);
    return sign(x) * y;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    let nt = f32(max(p.tiling, 1));
    let q = fract(uv * nt) * 2.0 - 1.0;
    let nb = max(p.branches, 3);
    let n = f32(nb);
    let theta = atan2(q.y, q.x) / TAU;                 // turns
    let k0 = i32(round(theta * n));
    // Gaussian blur (Blur Amount 1 = sigma 1.6 px at 256 px) on top of half a pixel of antialiasing, q units.
    let px = 2.0 * nt * pixel_uv();
    let blur = 2.0 * nt * 1.6 * max(p.blur_amount, 0.0) / 256.0;
    let k = 1.0 / (sqrt(2.0) * sqrt(blur * blur + 0.25 * px * px));
    var v = 0.0;
    let span = 2 + i32(ceil(p.randomness));
    for (var dk = -span; dk <= span; dk++) {
        let i = wrapi(k0 + dk, nb);
        let ang = (f32(i) + p.randomness * srnd(vec2i(i, 0), 1u) * 0.5) / n;
        let len = max(mix(p.size_max, mix(p.size_min, p.size_max, rnd(vec2i(i, 0), 2u)), p.size_variation), 1e-4);
        let dir = vec2f(cos(ang * TAU), sin(ang * TAU));
        let along = dot(q, dir);
        // Behind the centre the branch ends in a round cap (its base width).
        let perp = select(abs(dot(q, vec2f(-dir.y, dir.x))), length(q), along < 0.0);
        // A wedge fading from white at the centre to black at its tip, as narrow as it is dim.
        let g = clamp(1.0 - max(along, 0.0) / len, 0.0, 1.0);
        let hw = len * min(0.25, 0.23 * PI / n) * g;
        let across = 0.5 * (erf_as((hw - perp) * k) + erf_as((hw + perp) * k));
        v = max(v, g * across);
    }
    store(c, vec4f(v, v, v, 1.0));
}
