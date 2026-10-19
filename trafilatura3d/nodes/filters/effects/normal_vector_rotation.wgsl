@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dx = p.normal_format == 0;
    let t = load_normal(c);
    var n = fx_decode_normal(t, dx);
    let a = p.rotation_angle * load_rotation_map(c).r * 2.0 * PI;
    let xy = fx_rot(n.xy, a);
    store(c, vec4f(fx_encode_normal(vec3f(xy, n.z), dx), t.a));
}
