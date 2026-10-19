@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store_min(c, gray4(texel_lo(vec2i(0)).r));
    store_max(c, gray4(texel_hi(vec2i(0)).r));
}
