@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let y = load_position(c).g + (load_roughness(c).r - 0.5) * p.roughness_variation * 0.5;
    store(c, g4(contrast_level(1.0 - y, p.contrast, p.level) * load_mask(c).r));
}
