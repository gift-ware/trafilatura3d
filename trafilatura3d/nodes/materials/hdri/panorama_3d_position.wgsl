@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var v = ll_dir(uv_of(c, d));
    if (p.up_vector == 0) { v = vec3f(v.x, -v.z, v.y); }  // Z up, right-handed
    store(c, vec4f(v * 0.5 + 0.5, 1.0));
}
