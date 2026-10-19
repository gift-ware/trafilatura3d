// Lattice sites (ours): one texel per lattice cell, its site's offset from the cell's centre in cells (xy): a distance
// p.spread x u^p.edge (u uniform: 0.5 spreads them evenly over the disc, less pushes them out) at a random angle that
// Disorder turns; plus, with p.group > 0, an offset anywhere in a disc of p.group cells shared by the 2 x 2 cells of a
// group (clusters; the lattice must be even to tile). z: a random height in 0-1, with p.patchiness > 0
// times mix(1, a smooth random field over n / 8 cells, p.patchiness) (dense and empty regions; n a multiple of 8 to tile).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let h = hash22(c, p.seed_ + 13.7);
    let a = 6.28318531 * (h.y + p.disorder);
    var o = pow(h.x, p.edge) * p.spread * vec2f(cos(a), sin(a));
    if (p.group > 0.0) {
        let g = hash22(wrap(c / 2, max(d / 2, vec2i(1))), p.seed_ + 21.9);
        let b = 6.28318531 * (g.y + p.disorder);
        o += sqrt(g.x) * p.group * vec2f(cos(b), sin(b));
    }
    var z = hash21(c, p.seed_ + 29.3);
    if (p.patchiness > 0.0) {
        let m = max(d / 8, vec2i(1));
        let f = (vec2f(c) + 0.5) / 8.0 - 0.5;
        let i = vec2i(floor(f));
        let t = smoothstep(vec2f(0.0), vec2f(1.0), fract(f));
        let s = p.seed_ + 37.1;
        let a = mix(hash21(wrap(i, m), s), hash21(wrap(i + vec2i(1, 0), m), s), t.x);
        let b = mix(hash21(wrap(i + vec2i(0, 1), m), s), hash21(wrap(i + vec2i(1, 1), m), s), t.x);
        z *= mix(1.0, mix(a, b, t.y), p.patchiness);
    }
    store(c, vec4f(o, z, 1.0));
}
