@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let q = rot2((uv_of(c, d) - 0.5) * 2.0, p.rotation * TAU) / max(p.scale, 0.02);
    let dir = normalize(vec3f(q.x, -1.0, q.y));
    var v = sample_input(dir_ll(dir)).rgb;
    if (p.output_gamma == 0) { v = lin2srgb(v); }
    store(c, vec4f(v, 1.0));
}
