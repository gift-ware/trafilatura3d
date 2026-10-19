@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // Invert Position mirrors the position, it does not invert the result.
    let pos = select(p.position, 1.0 - p.position, p.invert_position != 0);
    store(c, gray4(histogram_scan(load_input(c).r, pos, p.contrast)));
}
