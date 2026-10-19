// Weave 2: a twill of slanted, slightly bent stitches, one on every point of the lattice
// (1/4, 0), (1/8, 1/8) per tiling unit. Across a stitch the height ramps up slowly from its left edge
// and drops steeply on its right, along it a flat-topped profile; the highest stitch wins over a low
// floor. Sizes in tiling units, tuned at Disorder 0.
const W2_A = vec2f(0.25, 0.0);
const W2_B = vec2f(0.125, 0.125);
const W2_ORIGIN = vec2f(0.05136, 0.14466);
const W2_ANGLE = -0.315;           // radians from vertical, clockwise on screen
const W2_LEFT = 0.06735;           // ramp up, left of the crest
const W2_RIGHT = 0.02502;          // drop, right of the crest
const W2_HALF_LENGTH = 0.16399;
const W2_BEND = 0.512;
const W2_FLOOR = 0.138;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let n = f32(max(p.scale, 1)) / 4.0;       // tiling units of four stitches per row
    let g = grid(uv, n, p.rotate_45_degrees);
    let dis = clamp(p.disorder, 0.0, 1.0);
    let cs = vec2f(cos(W2_ANGLE), sin(W2_ANGLE));
    let j0 = round((g.y - W2_ORIGIN.y) / W2_B.y);
    var v = W2_FLOOR;
    for (var dj = -3; dj <= 3; dj++) {
        let j = j0 + f32(dj);
        let i0 = round((g.x - W2_ORIGIN.x - j * W2_B.x) / W2_A.x);
        for (var di = -1; di <= 1; di++) {
            let ctr = W2_ORIGIN + (i0 + f32(di)) * W2_A + j * W2_B;
            // Random stitches keyed by their centre wrapped into the tile (repeats every n units).
            let key = vec2i(round((ctr - floor(ctr / n) * n) * 16.0));
            let o = g - ctr;
            let along = cs.y * o.x + cs.x * o.y;
            let across = cs.x * o.x - cs.y * o.y + W2_BEND * along * along
                         + dis * 0.3 * W2_LEFT * srnd(key, 501u);
            let up = pow(clamp(1.0 + across / W2_LEFT, 0.0, 1.0), 2.509);
            let down = pow(clamp(1.0 - across / W2_RIGHT, 0.0, 1.0), 0.737);
            let len = W2_HALF_LENGTH * (1.0 - 0.3 * dis * rnd(key, 502u));
            let h = clamp(1.0 - pow(abs(along / len), 3.699), 0.0, 1.0);
            let amp = 0.908 * (1.0 - 0.5 * dis * rnd(key, 503u));
            v = max(v, W2_FLOOR + amp * select(down, up, across < 0.0) * h);
        }
    }
    store(c, vec4f(v, v, v, 1.0));
}
