@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = trace(c);
    var n = ray_normal(fx_uv(c));
    if (length(t.yzw) > 1e-6) { n = normalize(t.yzw); }
    store(c, vec4f(fx_encode_normal(n, p.normal_format == 0), 1.0));
}
