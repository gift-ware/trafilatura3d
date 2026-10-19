// Small spots (ours): two lattices of p.cells and 2 x p.cells cells across the image; every cell holds a Poisson
// number of gaussian bells (mean 1, at most 4) at uniform random points (a Poisson process: clumps and empty places),
// each sigma drawn per axis between p.sigma_lo and p.sigma_hi cells (round and elongated spots); Disorder moves each
// bell along a circle of 0.15 cells. The image is p.gain x the sum of the bells, clamped. The bells reach 3 sigma
// <= 0.7 cells: the 3 x 3 cells around a pixel hold every bell.

fn layer(uv: vec2f, n: i32, s: f32) -> f32 {
    let x = uv * f32(n);
    let i = vec2i(floor(x));
    let m = vec2i(n, n);
    var sum = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var k = -1; k <= 1; k++) {
            let q = i + vec2i(k, j);
            let w = wrap(q, m);
            let u = hash21(w, s);
            // Poisson(1) by its cumulative probabilities
            let count = select(0, 1, u > 0.3679) + select(0, 1, u > 0.7358) + select(0, 1, u > 0.9197)
                      + select(0, 1, u > 0.9810);
            for (var t = 0; t < count; t++) {
                let f = f32(t) * 7.31;
                let h = hash22(w, s + 1.9 + f);
                let g = hash22(w, s + 4.7 + f);
                var at = vec2f(q) + h;
                if (p.disorder != 0.0) {
                    let a = 6.28318531 * fract(g.x * 13.0 + g.y * 7.0);
                    let b = a + 6.28318531 * p.disorder;
                    at += 0.15 * vec2f(cos(b) - cos(a), sin(b) - sin(a));
                }
                let sg = mix(vec2f(p.sigma_lo), vec2f(p.sigma_hi), g);
                let e = (x - at) / sg;
                sum += exp(-0.5 * dot(e, e));
            }
        }
    }
    return sum;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let v = clamp(p.gain * (layer(uv, p.cells, p.seed_ + 3.1) + layer(uv, 2 * p.cells, p.seed_ + 11.7)), 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
