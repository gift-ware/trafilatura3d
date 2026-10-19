// Crystal 1 (ours): two Worley fields from two random streams, each the distance to the nearest of one jittered
// point per cell (Disorder turns the points around their cells); the output is the difference of the two
// distances, scaled by CR_GAIN: dark along the lines where the two point sets are equally near, bright facets
// where one is much nearer. A cell per CELLS of Scale (the preview at the default Scale 32 has 14 cells across);
// CR_GAIN fitted on the preview's mean and spread.

const CELLS = 0.4375;
const CR_GAIN = 0.75;
const CR_GAMMA = 1.1;

fn nearest(x: vec2f, per: vec2i, s: f32, dis: f32) -> f32 {
    let i = vec2i(floor(x));
    var best = 1e9;
    for (var j = -2; j <= 2; j++) {
        for (var ii = -2; ii <= 2; ii++) {
            let q = i + vec2i(ii, j);
            let rest = hw2(q, per, s) - 0.5;
            let point = vec2f(q) + 0.5 + rot2(rest, dis);
            best = min(best, dot(x - point, x - point));
        }
    }
    return best;  // squared
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per = max(vec2i(round(f32(max(p.scale, 1)) * CELLS * asp)), vec2i(1));
    let x = uv_of(c) * vec2f(per);
    let v = pow(clamp(CR_GAIN * abs(nearest(x, per, p.seed_, p.disorder) - nearest(x, per, p.seed_ + 1.0, p.disorder)),
                      0.0, 1.0), CR_GAMMA);
    store(c, vec4f(v, v, v, 1.0));
}
