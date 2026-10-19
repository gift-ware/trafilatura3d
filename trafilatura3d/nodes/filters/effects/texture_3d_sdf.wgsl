// Final pass: distance to the nearest surface voxel, in unit-cube units, signed by the mask.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let res = fx_slice_res(d);
    let v = fx_texel_voxel(c, res);
    let vs = fx_voxel_size(res);
    let s = texel_seeds(c);
    let inside = load_mask_input(c).r >= p.threshold;
    var dist = 2.0;  // no surface at all: far
    if (s.w > 0.0) { dist = length((s.xyz - vec3f(v)) * vs); }
    let half_voxel = 0.5 * min(vs.x, min(vs.y, vs.z));
    var o = max(dist - half_voxel, 0.0);
    if (inside) { o = select(0.0, -(dist + half_voxel), p.output == 1); }
    if (s.w <= 0.0 && inside) { o = select(0.0, -2.0, p.output == 1); }
    store(c, fx_gray(o));
}
