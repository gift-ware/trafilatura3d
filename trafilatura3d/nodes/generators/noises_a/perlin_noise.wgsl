// Perlin noise (ours): gradient noise as a sum of surflets. On every cell of the Scale grid a random unit gradient g
// at the cell's centre; a point x gets sum over the 3x3 nearest cells of dot(g, d) x (1 - |d|^2 / R^2)^4, d = x minus
// the centre, R = PN_RADIUS cells (R <= 1.5 keeps every surflet that reaches x within those 3x3 cells). Disorder x
// speed turns every gradient by that many turns (1 loops). The square tile repeats across a non-square image
// (Non-square expansion), Tile offset shifts it. R and the gain are fitted on the output's spectrum and spread.

const PN_RADIUS = 1.5;
const PN_GAIN = 0.525;

fn pn_grad(q: vec2i, per: vec2i, turn: f32) -> vec2f {
    let a = (hw(q, per, p.seed_ + 0.37) + turn) * TAU;
    return vec2f(cos(a), sin(a));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = max(p.scale, 1);
    let per = vec2i(n);
    let x = (uv_of(c) * aspect(p.non_square_expansion) + p.tile_offset) * f32(n);
    let turn = p.disorder * p.disorder_speed;
    let i = vec2i(floor(x));
    var s = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = i + vec2i(dx, dy);
            let e = x - vec2f(q) - 0.5;
            let t = max(1.0 - dot(e, e) / (PN_RADIUS * PN_RADIUS), 0.0);
            let t2 = t * t;
            s += dot(pn_grad(q, per, turn), e) * t2 * t2;
        }
    }
    let v = clamp(0.5 + PN_GAIN * s, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
