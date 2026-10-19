// One jump flooding pass for two seed sets (xy and zw, -1 = none known): keep the nearest seen at +-step around.
// UV space does not wrap around.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (any(c >= d)) { return; }
    let me = vec2f(c);
    var best = texel_seeds(c);
    var ba = select(1e20, distance(me, best.xy), best.x >= 0.0);
    var bb = select(1e20, distance(me, best.zw), best.z >= 0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let q = c + vec2i(i, j) * p.step_;
            if (any(q < vec2i(0)) || any(q >= d) || (i == 0 && j == 0)) { continue; }
            let s = textureLoad(src_seeds, q, 0);
            if (s.x >= 0.0) {
                let da = distance(me, s.xy);
                if (da < ba) { ba = da; best.x = s.x; best.y = s.y; }
            }
            if (s.z >= 0.0) {
                let db = distance(me, s.zw);
                if (db < bb) { bb = db; best.z = s.z; best.w = s.w; }
            }
        }
    }
    store(c, best);
}
