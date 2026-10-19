@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let up = vec3f(0.0, 1.0, 0.0);
    let a = p.direction_angle * TAU;
    let r = rot_axis(up, p.post_rotation_around_pole * TAU) * rot_axis(vec3f(cos(a), 0.0, sin(a)), p.rotation * TAU)
        * rot_axis(up, p.pre_rotation_around_pole * TAU);
    let src = transpose(r) * ll_dir(uv_of(c, d));  // output direction -> where it came from
    store(c, sample_input(dir_ll(src)));
}
