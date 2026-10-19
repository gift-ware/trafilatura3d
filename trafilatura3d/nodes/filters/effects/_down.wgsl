// 2x2 average towards half the size.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = size_src();
    var acc = vec4f(0.0);
    var n = 0.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let q = c * 2 + vec2i(i, j);
            if (q.x < s.x && q.y < s.y) { acc += texel_src(q); n += 1.0; }
        }
    }
    store_out(c, acc / max(n, 1.0));
}
