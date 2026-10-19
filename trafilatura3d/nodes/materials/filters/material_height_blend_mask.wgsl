@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store(c, g4(hb_mask(load_height_fg(c).r, load_height_bg(c).r, p.height_offset, p.contrast, p.mode, p.opacity)));
}
