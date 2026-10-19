@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store(c, vec4f(load_r(c).r, load_g(c).r, load_b(c).r, load_a(c).r));
}
