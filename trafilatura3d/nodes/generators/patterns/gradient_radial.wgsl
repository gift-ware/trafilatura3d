@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let fs = frame_size(p.non_square_expansion);
    let a = p.point_1 * fs;
    let ab = p.point_2 * fs - a;
    let r = length(uv - a) / max(length(ab), 1e-10);
    var v = max(1.0 - r, 0.0);
    if (p.shape == 1) { v = sqrt(max(1.0 - r * r, 0.0)); }
    store(c, vec4f(v, v, v, 1.0));
}
