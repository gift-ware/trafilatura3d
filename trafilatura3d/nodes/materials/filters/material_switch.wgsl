@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store_$ch(c, out_$ch(c, select(load_$ch_1(c), load_$ch_2(c), p.switch_ != 0)));
}
