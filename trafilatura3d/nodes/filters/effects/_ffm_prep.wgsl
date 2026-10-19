// Per shape pixel: its box's larger side and its negation (max reduction -> largest, smallest).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_flood_fill_bbox(c);
    let s = max(t.z, t.w);
    store_out(c, select(vec4f(-1.0), vec4f(s, -s, 0.0, 0.0), t.z > 0.0 && t.w > 0.0));
}
