@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var l = luma(load_basecolor(c).rgb);
    if (p.invert_generated_relief != 0) { l = 1.0 - l; }
    store(c, g4(l));
}
