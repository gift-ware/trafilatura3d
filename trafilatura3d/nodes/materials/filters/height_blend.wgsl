@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let h = hb_heights(load_height_top(c).r, load_height_bottom(c).r, p.height_offset, p.mode, load_mask(c).r);
    let m = hb_mask_of(h, p.contrast, p.opacity);
    store_height(c, g4(mix(h.y, h.x, m)));
    store_mask(c, g4(m));
}
