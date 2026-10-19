@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let white = load_mask_input(c).r >= 0.5;
    let s = select(vec2f(-1.0), vec2f(c), white);
    store_seeds(c, jfa_pack(vec4f(s.x, 0.0, 0.0, s.y)));
}
