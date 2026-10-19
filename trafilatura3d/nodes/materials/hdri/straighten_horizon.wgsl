@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let up = vec3f(0.0, 1.0, 0.0);
    var r = mat3x3f(vec3f(1.0, 0.0, 0.0), up, vec3f(0.0, 0.0, 1.0));
    if (p.enable_horizon_correction != 0) {
        var n = cross(ll_dir(p.horizon_point_1_position), ll_dir(p.horizon_point_2_position));
        if (length(n) > 1e-5) {
            n = normalize(n);
            if (n.y < 0.0) { n = -n; }
            let axis = cross(n, up);
            if (length(axis) > 1e-5) { r = rot_axis(axis, acos(clamp(n.y, -1.0, 1.0))); }  // horizon normal -> up
        }
    }
    let src = transpose(r) * (transpose(rot_axis(up, p.horizontal_rotation * TAU)) * ll_dir(uv_of(c, d)));
    store(c, sample_input(dir_ll(src)));
}
