// One step of an inclusive prefix sum along x or y: add the values j * `step` pixels back, 0 < j < p.radix
// (radix 2: Hillis-Steele).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var v = texel_src(c);
    let ax = select(vec2i(1, 0), vec2i(0, 1), p.vertical == 1);
    for (var j = 1; j < p.radix; j++) {
        let q = c - ax * (j * p.step);
        if (q.x < 0 || q.y < 0) { break; }
        v += texel_src(q);
    }
    store_out(c, v);
}
