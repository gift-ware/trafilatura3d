@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let m = equalizer_mask(load_$in(c).rgb, load_avg(c).rgb, load_mask_input(c).r);
    store_mask(c, vec4f(m, m, m, 1.0));
}
