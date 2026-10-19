@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_source(c);
    let q = uv_at(c) + p.offset;
    var s = vec4f(0.0);
    if (!off_image(q)) { s = sample_source(q); }
    store(c, mix(v, s, clamp(load_mask(c).r, 0.0, 1.0) * s.a));
}
