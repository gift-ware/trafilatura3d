@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    var pos = vec3f(uv, 0.0);
    if (POSITION_CONNECTED) { pos = load_position(c).rgb; }
    var a = p.point_1_world;
    var b = p.point_2_world;
    if (p.points_position_mode == 0) {
        a = vec3f(p.point_1_uv, 0.0);
        b = vec3f(p.point_2_uv, 0.0);
        if (POSITION_CONNECTED) {
            a = sample_position(p.point_1_uv).rgb;
            b = sample_position(p.point_2_uv).rgb;
        }
    }
    let ab = b - a;
    var t = clamp(dot(pos - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    let k = max(1.0 - clamp(p.contrast, 0.0, 1.0), 1e-3);
    t = clamp((t - 0.5) / k + 0.5, 0.0, 1.0);
    store(c, vec4f(t, t, t, 1.0));
}
