// Plasma (ours): two smooth fields from two sets of sites on the same lattice (`sites_a`, `sites_b`, from
// lattice_sites.wgsl), each the sum of gaussian bells of sigma p.sigma cells on its sites, centred on its mean and
// scaled by p.contrast around 0.5, clamped to 0-1 (flat where saturated); the output is
// (p.gain x |A - B|)^p.gamma, clamped: bright where the fields differ, dark veins where they cross. The 7 x 7 cells
// around a pixel hold every bell that reaches it (3 sigma) while the sites stay within 3.5 - 3 sigma cells.
const R = 3;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_sites_a();
    let x = (vec2f(c) + 0.5) / vec2f(d) * vec2f(n);
    let i = vec2i(floor(x));
    var cols: array<i32, 7>;  // the wrapped columns, once (integer modulo is slow)
    for (var k = -R; k <= R; k++) { cols[k + R] = wrap1(i.x + k, n.x); }
    let w = -0.5 / (p.sigma * p.sigma);
    var sa = 0.0;
    var sb = 0.0;
    for (var j = -R; j <= R; j++) {
        let row = wrap1(i.y + j, n.y);
        for (var k = -R; k <= R; k++) {
            let q = vec2i(cols[k + R], row);
            let e = x - (vec2f(i + vec2i(k, j)) + 0.5);
            let ea = e - textureLoad(src_sites_a, q, 0).xy;
            let eb = e - textureLoad(src_sites_b, q, 0).xy;
            sa += exp(w * dot(ea, ea));
            sb += exp(w * dot(eb, eb));
        }
    }
    let mean = 6.28318531 * p.sigma * p.sigma;  // one bell per cell
    let a = clamp(0.5 + p.contrast * (sa - mean), 0.0, 1.0);
    let b = clamp(0.5 + p.contrast * (sb - mean), 0.0, 1.0);
    let v = pow(clamp(p.gain * abs(a - b), 0.0, 1.0), p.gamma);
    store(c, vec4f(v, v, v, 1.0));
}
