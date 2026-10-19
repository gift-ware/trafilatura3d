// 2x2 max reduction (per channel) towards a 1x1 texture.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = size_src();
    var m = vec4f(-3.0e38);
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let q = c * 2 + vec2i(i, j);
            if (q.x < s.x && q.y < s.y) { m = max(m, textureLoad(src_src, q, 0)); }
        }
    }
    store_out(c, m);
}
