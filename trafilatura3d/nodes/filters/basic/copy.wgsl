@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var v = load_input(c);
    if (OUT_OUTPUT_GRAY) { v = gray4(v.r); }
    store(c, v);
}
