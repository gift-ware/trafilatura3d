@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let fs = frame_size(p.non_square_expansion);
    let a = p.point_1 * fs;
    let ab = p.point_2 * fs - a;
    let e = uv - a;
    let v = fract((atan2(e.y, e.x) - atan2(ab.y, ab.x)) / TAU + 1.0 + p.angle);
    store(c, vec4f(v, v, v, 1.0));
}
