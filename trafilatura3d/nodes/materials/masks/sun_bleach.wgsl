@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = wsn(load_normal_world_space(c).rgb);
    var v = n.y * 0.5 + 0.5;
    v *= mix(1.0, load_ambient_occlusion(c).r, p.occlusion);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
