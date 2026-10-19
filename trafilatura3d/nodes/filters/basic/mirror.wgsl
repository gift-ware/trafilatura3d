@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var q = uv_at(c);
    if (p.mode == 0) {
        let a = p.axis_x_offset;
        if ((q.x > a) != (p.invert_axis_x != 0)) { q.x = 2.0 * a - q.x; }
    } else if (p.mode == 1) {
        let a = p.axis_y_offset;
        if ((q.y > a) != (p.invert_axis_y != 0)) { q.y = 2.0 * a - q.y; }
    } else {
        let right = p.corner_type == 1 || p.corner_type == 3;
        let bottom = p.corner_type >= 2;
        if ((q.x > 0.5) != right) { q.x = 1.0 - q.x; }
        if ((q.y > 0.5) != bottom) { q.y = 1.0 - q.y; }
    }
    // Texel-exact: reflect the pixel index so a mirrored edge repeats nothing.
    let s = vec2f(size_input());
    store(c, texel_input(vec2i(floor(q * s))));
}
