@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c).xyz * 2.0 - 1.0;
    let l = length(v);
    store(c, fx_gray(select(1.0, clamp(v.z / l, 0.0, 1.0), l > 1e-6)));
}
