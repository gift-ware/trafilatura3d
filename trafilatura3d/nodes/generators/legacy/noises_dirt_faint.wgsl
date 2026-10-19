// Faint dirt (ours): sparse convolution on three grids of p.cells, 2 p.cells and 4 p.cells cells across (weights 1,
// p.weights.x, p.weights.y). Each cell holds one tent bump (1 - |x| / r)(1 - |y| / r), r = p.radius cells, at a random
// point of the cell, with a random height u in [0, 1) less p.bias: the sum is signed. Kept above zero times p.gain,
// raised to p.power.
fn grid(x: vec2f, n: i32, s: f32) -> f32 {
    let q = vec2i(floor(x));
    var f = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let cell = q + vec2i(i, j);
            let w = wrap(cell, vec2i(n));
            let at = vec2f(cell) + hash22(w, s);
            let o = abs(x - at) / p.radius;
            let b = max(1.0 - o.x, 0.0) * max(1.0 - o.y, 0.0);
            f += b * (hash21(w, s + 5.3) - p.bias);
        }
    }
    return f;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let n = max(p.cells, 1);
    let s = p.seed_ * 0.618 + 3.1;
    let f = grid(uv * f32(n), n, s) + p.weights.x * grid(uv * f32(2 * n), 2 * n, s + 11.0)
        + p.weights.y * grid(uv * f32(4 * n), 4 * n, s + 23.0);
    let v = pow(clamp(p.gain * f, 0.0, 1.0), p.power);
    store(c, vec4f(v, v, v, 1.0));
}
