// One jump flooding pass for both seed sets: keep the nearest seed seen at +-step around (packed: _jfa_lib.wgsl).
// `p.step` 0: the last pass, the seeds unpacked into the 32 bit result the nodes read.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let me = vec2f(c);
    var best = jfa_unpack(texel_seeds(c));
    if (p.step == 0) { store_seeds(c, best); return; }
    var bw = select(1e20, length(fx_wrap_delta(me, best.xy)), best.x >= 0.0);
    var bb = select(1e20, length(fx_wrap_delta(me, best.zw)), best.z >= 0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            if (i == 0 && j == 0) { continue; }
            let q = c + vec2i(i, j) * p.step;
            if (!(tile_h() || (q.x >= 0 && q.x < d.x)) || !(tile_v() || (q.y >= 0 && q.y < d.y))) { continue; }
            let s = jfa_unpack(texel_seeds(q));
            if (s.x >= 0.0) {
                let dw = length(fx_wrap_delta(me, s.xy));
                if (dw < bw) { bw = dw; best.x = s.x; best.y = s.y; }
            }
            if (s.z >= 0.0) {
                let db = length(fx_wrap_delta(me, s.zw));
                if (db < bb) { bb = db; best.z = s.z; best.w = s.w; }
            }
        }
    }
    store_seeds(c, jfa_pack(best));
}
