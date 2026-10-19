// Flood fill: each white pixel joins its shape with its white neighbours above and on the left (4-connected,
// across the tiled edges; the left one inside the image was joined by the first pass).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let i = i32(ff_index(c));
    if (!ff_white(i)) { return; }
    if (c.x == 0) {
        let l = ff_step(c, vec2i(-1, 0));
        if (ff_white(l)) { ff_union(u32(i), u32(l)); }
    }
    let u = ff_step(c, vec2i(0, -1));
    if (ff_white(u)) { ff_union(u32(i), u32(u)); }
}
