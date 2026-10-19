// Surface voxels (inside, with an outside 6-neighbour) seed themselves; the others have none (-1).
fn inside(v: vec3i, res: vec2i) -> bool {
    if (any(v < vec3i(0)) || v.x >= res.x || v.y >= res.y || v.z >= fx_slices()) { return false; }
    return load_mask_input(fx_voxel_texel(v, res)).r >= p.threshold;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let res = fx_slice_res(d);
    let v = fx_texel_voxel(c, res);
    var surface = false;
    if (inside(v, res)) {
        let n = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0),
                                vec3i(0, 0, 1), vec3i(0, 0, -1));
        for (var k = 0; k < 6; k++) { if (!inside(v + n[k], res)) { surface = true; } }
    }
    store_seeds(c, select(vec4f(-1.0), vec4f(vec3f(v), 1.0), surface));
}
