@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t1 = load_normal_1(c).xyz;
    let t2 = load_normal_2(c).xyz;
    var o: vec3f;
    if (p.technique == 0) {
        // Encoded XY added (Add Sub), encoded Z multiplied, normalised.
        let v = vec3f(clamp(t1.xy + t2.xy - 0.5, vec2f(0.0), vec2f(1.0)), t1.z * t2.z) * 2.0 - 1.0;
        o = select(vec3f(0.0, 0.0, 1.0), normalize(v), length(v) > 1e-6) * 0.5 + 0.5;
    } else if (p.technique == 1) {
        // Encoded XY added, the lower Z kept; not normalised.
        o = vec3f(clamp(t1.xy + t2.xy - 0.5, vec2f(0.0), vec2f(1.0)), min(t1.z, t2.z));
    } else {
        o = fx_rnm(normalize(t1 * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5)),
                   normalize(t2 * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5))) * 0.5 + 0.5;
    }
    store(c, vec4f(o, 1.0));
}
