@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c).rgb;
    let hi = max(v.r, max(v.g, v.b));
    let lo = min(v.r, min(v.g, v.b));
    var o = vec3f(luma709(clamp(v, vec3f(0.0), vec3f(1.0))) * 0.5);
    if (hi > 1.0) {
        let t = clamp((hi - 1.0) / max(p.range_max - 1.0, 1e-4), 0.0, 1.0);
        o = vec3f(1.0, t, 0.0);
    } else if (lo < 0.0) {
        let t = clamp(lo / min(p.range_min, -1e-4), 0.0, 1.0);
        o = vec3f(0.0, t, 1.0);
    }
    store(c, vec4f(o, 1.0));
}
