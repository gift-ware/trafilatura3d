// 3D jump flooding: keep the nearest seed among the 26 neighbours at +-step (no wrap-around).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let res = fx_slice_res(d);
    let v = fx_texel_voxel(c, res);
    let vs = fx_voxel_size(res);
    var best = texel_seeds(c);
    var bd = select(1e20, length((best.xyz - vec3f(v)) * vs), best.w > 0.0);
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let q = v + vec3i(i, j, k) * p.step;
                if (any(q < vec3i(0)) || q.x >= res.x || q.y >= res.y || q.z >= fx_slices()) { continue; }
                let s = texel_seeds(fx_voxel_texel(q, res));
                if (s.w <= 0.0) { continue; }
                let dd = length((s.xyz - vec3f(v)) * vs);
                if (dd < bd) { bd = dd; best = s; }
            }
        }
    }
    store_seeds(c, best);
}
