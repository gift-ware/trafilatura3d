@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // One Levels to the output range centred on Position, Range / 2 each side as far as it fits in 0..1.
    let h = min(p.range * 0.5, min(p.position, 1.0 - p.position));
    store(c, gray4(clamp(mix(p.position - h, p.position + h, load_input(c).r), 0.0, 1.0)));
}
