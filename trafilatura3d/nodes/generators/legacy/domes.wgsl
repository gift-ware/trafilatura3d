// Domes (ours): on the lattice of `sites` (one texel per cell: the site's offset from the cell's centre, from
// lattice_sites.wgsl) a paraboloid 1 - (r / p.radius)^2 on every site; the image is their Max, black where none
// reaches. The 7 x 7 cells around a pixel hold every site that reaches it while spread + radius <= 3.5.
const R = 3;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_sites();
    let x = (vec2f(c) + 0.5) / vec2f(d) * vec2f(n);
    let i = vec2i(floor(x));
    var cols: array<i32, 7>;  // the wrapped columns, once (integer modulo is slow)
    for (var k = -R; k <= R; k++) { cols[k + R] = wrap1(i.x + k, n.x); }
    var best = 1e9;
    for (var j = -R; j <= R; j++) {
        let row = wrap1(i.y + j, n.y);
        for (var k = -R; k <= R; k++) {
            let e = x - (vec2f(i + vec2i(k, j)) + 0.5 + textureLoad(src_sites, vec2i(cols[k + R], row), 0).xy);
            best = min(best, dot(e, e));
        }
    }
    let v = max(1.0 - best / (p.radius * p.radius), 0.0);
    store(c, vec4f(v, v, v, 1.0));
}
