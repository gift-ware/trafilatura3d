// Blobs (ours): on the lattice of `sites` (one texel per cell: the site's offset from the cell's centre, from
// lattice_sites.wgsl) a gaussian bell of sigma p.sigma cells on every site, its height 1 or, with p.random_height, the
// site's random height; the image is p.lift + p.gain x their sum, its contrast around the mean times p.contrast.
// p.flip mirrors the image through its centre. The (2 p.reach + 1)^2 cells around a pixel (p.reach 2 or 3) hold
// every bell that reaches it (to 3 sigma) while the sites' offset + 3 sigma <= p.reach + 0.5.
const R = 3;  // the widest search

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_sites();
    var uv = (vec2f(c) + 0.5) / vec2f(d);
    if (p.flip != 0) { uv = 1.0 - uv; }
    let x = uv * vec2f(n);
    let i = vec2i(floor(x));
    var cols: array<i32, 7>;  // the wrapped columns, once (integer modulo is slow)
    let r = clamp(p.reach, 0, R);
    for (var k = -r; k <= r; k++) { cols[k + R] = wrap1(i.x + k, n.x); }
    let w = -0.5 / (p.sigma * p.sigma);
    var sum = 0.0;
    for (var j = -r; j <= r; j++) {
        let row = wrap1(i.y + j, n.y);
        for (var k = -r; k <= r; k++) {
            let s = textureLoad(src_sites, vec2i(cols[k + R], row), 0);
            let e = x - (vec2f(i + vec2i(k, j)) + 0.5 + s.xy);
            sum += select(1.0, s.z, p.random_height != 0) * exp(w * dot(e, e));
        }
    }
    // the sum's mean: one bell per cell, its height 1 or 0.5 on average
    let mean = 6.28318531 * p.sigma * p.sigma * select(1.0, 0.5, p.random_height != 0);
    let v = p.lift + p.gain * (mean + p.contrast * (sum - mean));
    store(c, vec4f(v, v, v, 1.0));
}
