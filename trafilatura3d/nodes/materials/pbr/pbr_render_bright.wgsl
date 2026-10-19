@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_hdr(c).rgb * exp2(p.exposure);
    store(c, vec4f(max(v - vec3f(p.bloom_threshold), vec3f(0.0)), 1.0));
}
