// Crystal facets (ours): two fields of domes on the same lattice from two sets of sites (`sites_a`, `sites_b`, from
// lattice_sites.wgsl), each the Max of paraboloids p.height x (1 - (r / p.radius)^2) (black where none reaches). The
// output is p.gain x |A - B|, clamped: dark creases where the two fields cross, bright facets where one dominates;
// `product` is A x B (the warp's map). The 7 x 7 cells around a pixel hold every site that reaches it while
// spread + radius <= 3.5.
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
    var ba = 1e9;
    var bb = 1e9;
    for (var j = -R; j <= R; j++) {
        let row = wrap1(i.y + j, n.y);
        for (var k = -R; k <= R; k++) {
            let q = vec2i(cols[k + R], row);
            let e = x - (vec2f(i + vec2i(k, j)) + 0.5);
            let ea = e - textureLoad(src_sites_a, q, 0).xy;
            let eb = e - textureLoad(src_sites_b, q, 0).xy;
            ba = min(ba, dot(ea, ea));
            bb = min(bb, dot(eb, eb));
        }
    }
    let r2 = p.radius * p.radius;
    let a = p.height * max(1.0 - ba / r2, 0.0);
    let b = p.height * max(1.0 - bb / r2, 0.0);
    let v = clamp(p.gain * abs(a - b), 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
    store_product(c, vec4f(a * b, a * b, a * b, 1.0));
}
