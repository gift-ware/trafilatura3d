@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    store_r(c, gray4(v.r));
    store_g(c, gray4(v.g));
    store_b(c, gray4(v.b));
    store_a(c, gray4(v.a));
}
