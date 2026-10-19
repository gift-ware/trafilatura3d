// Flat cells (ours): the region nearest each site of `sites` (lattice_sites.wgsl, the sites of Cells Domes) takes one
// gray: a level k / (p.colors - 1), k random in 1 .. p.colors - 1, times an attenuation read at the site from a smooth
// random field (value noise on a p.grid x p.grid lattice over the tile), cut to ((f - p.lo) / (p.hi - p.lo))^p.power:
// dark and bright neighbourhoods spanning a few cells, black where the field is below p.lo. Hard edges, as the cells
// of the domes meet. The 7 x 7 cells around a pixel hold the nearest site while the sites stay within 2.5 cells.
const R = 3;

fn field(uv: vec2f) -> f32 {
    let m = vec2i(max(p.grid, 1));
    let f = uv * vec2f(m);
    let i = vec2i(floor(f));
    let t = smoothstep(vec2f(0.0), vec2f(1.0), fract(f));
    let s = p.seed_ + 61.3;
    let a = mix(hash21(wrap(i, m), s), hash21(wrap(i + vec2i(1, 0), m), s), t.x);
    let b = mix(hash21(wrap(i + vec2i(0, 1), m), s), hash21(wrap(i + vec2i(1, 1), m), s), t.x);
    return mix(a, b, t.y);
}

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
    var cell = vec2i(0);
    var site = vec2f(0.0);
    for (var j = -R; j <= R; j++) {
        let row = wrap1(i.y + j, n.y);
        for (var k = -R; k <= R; k++) {
            let q = vec2i(cols[k + R], row);
            let s = vec2f(i + vec2i(k, j)) + 0.5 + textureLoad(src_sites, q, 0).xy;
            let dd = dot(x - s, x - s);
            if (dd < best) { best = dd; cell = q; site = s; }
        }
    }
    let levels = f32(max(p.colors - 1, 1));
    let gray = max(ceil(hash21(cell, p.seed_ + 53.9) * levels), 1.0) / levels;
    let a = clamp((field(site / vec2f(n)) - p.lo) / (p.hi - p.lo), 0.0, 1.0);
    let v = gray * pow(a, p.power);
    store(c, vec4f(v, v, v, 1.0));
}
