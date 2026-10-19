// Cells mask (ours): white cells, the regions nearest the sites of `sites` (lattice_sites.wgsl, the sites of Cells
// Domes), and black walls along their borders (the bisector of the two nearest sites s1, s2), p.width cells plus
// p.width_px pixels wide, with a pixel of antialiasing. The 7 x 7 cells around a pixel hold the two nearest sites
// while the sites stay within 2.5 cells.
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
    var d1 = 1e9;
    var d2 = 1e9;
    var s1 = vec2f(0.0);
    var s2 = vec2f(0.0);
    for (var j = -R; j <= R; j++) {
        let row = wrap1(i.y + j, n.y);
        for (var k = -R; k <= R; k++) {
            let s = vec2f(i + vec2i(k, j)) + 0.5 + textureLoad(src_sites, vec2i(cols[k + R], row), 0).xy;
            let dd = dot(x - s, x - s);
            if (dd < d1) { d2 = d1; s2 = s1; d1 = dd; s1 = s; } else if (dd < d2) { d2 = dd; s2 = s; }
        }
    }
    let e = (d2 - d1) / (2.0 * max(length(s2 - s1), 1e-4));  // distance to the border, in cells
    let px = f32(n.x) / f32(d.x);                              // a pixel, in cells
    let v = clamp((e - 0.5 * (p.width + p.width_px * px)) / px + 0.5, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
