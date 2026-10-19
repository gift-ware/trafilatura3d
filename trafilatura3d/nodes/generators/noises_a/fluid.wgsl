// Drips: one per cell of a periodic grid, a round bright head with a trail above it darkening
// upwards, shaded as a tube, overlaid with Max. Scale repeats the whole pattern: the
// drips (width, length, wobble) are given at the default Scale and shrink with the grid.
// Internal, all 0 for Fluid: flow (the drips fall `flow` cells, each a little more or less: FLOW_SPREAD x
// flow), jitter (each drip moved by up to half `jitter` cells per axis), both within its cell (a drip leaving
// its cell makes way for a new one, so the density stays even), and gaps (the share of cells without a drip).
const FLOW_SPREAD: f32 = 0.5;
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per = max(vec2i(round(f32(max(p.scale, 1)) * 8.0 * asp)), vec2i(1));
    let cell = asp / vec2f(per);
    let q = uv_of(c) * asp - vec2f(0.0, p.flow * cell.y);
    let i = vec2i(floor(q / cell));
    let z = DEFAULT_SCALE / f32(max(p.scale, 1));
    let w = 0.05 * z * clamp(p.pattern_size.x, 0.02, 1.0);  // half width, tile units
    let max_len = 0.25 * z * clamp(p.pattern_size.y, 0.02, 2.0);  // up to 2: Fluid (Legacy)'s longest
    let wob_amp = 0.02 * z * p.warp_intensity;
    let rx = i32(ceil((w + wob_amp) / cell.x)) + 1;
    let ry = i32(ceil(max_len / cell.y)) + 1;
    var v = 0.0;
    for (var j = -1; j <= ry; j++) {
        for (var ii = -rx; ii <= rx; ii++) {
            let qc = i + vec2i(ii, j);
            let g = hw2(qc, per, p.seed_ + 3.0);
            // Flow and jitter: a drip leaving its cell makes way for a new one (position, length, brightness).
            let h0 = hw2(qc, per, p.seed_);
            let jo = (hw2(qc, per, p.seed_ + 7.0) - 0.5) * p.jitter;
            let moved = vec2f(h0.x + jo.x, h0.y + jo.y + p.flow * FLOW_SPREAD * (g.x - 0.5));
            let k = p.seed_ + 13.0 * floor(moved.y) + 31.0 * floor(moved.x);
            let h = vec2f(select(fract(moved.x), hw2(qc, per, k).x, k != p.seed_), fract(moved.y));
            if (hw(qc, per, k + 11.0) < p.gaps) { continue; }
            // Disorder: each drip falls through its cell at its own whole speed (loops at 1).
            let fall = fract(h.y + p.disorder * floor(1.0 + g.x * 3.0));
            let head = (vec2f(qc) + vec2f(h.x, fall)) * cell;
            let len = max_len * (0.3 + 0.7 * select(g.y, hw2(qc, per, k + 3.0).y, k != p.seed_));
            let dy = head.y - q.y;  // > 0 above the head
            // Sideways wobble growing up the trail.
            let wob = wob_amp * sin(TAU * (dy * 6.0 / z + g.x)) * clamp(dy / len, 0.0, 1.0);
            let dx = (q.x - head.x - wob) / w;
            var s = 0.0;
            if (dy >= 0.0 && dy <= len) {
                let t = 1.0 - dy / len;
                s = sqrt(max(1.0 - dx * dx, 0.0)) * (0.1 + 0.9 * t * sqrt(t));
            } else if (dy < 0.0) {
                let r = dx * dx + (dy / w) * (dy / w);
                s = sqrt(max(1.0 - r, 0.0));
            }
            v = max(v, s * (0.75 + 0.25 * hw(qc, per, k + 5.0)));
        }
    }
    store(c, vec4f(v, v, v, 1.0));
}
