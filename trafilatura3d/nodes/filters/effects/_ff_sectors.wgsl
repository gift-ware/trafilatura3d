// Flood fill: each white pixel marks in its shape's values the sectors it covers (only the first pixel of a sector
// along a run marks it: few atomics on a large shape).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let i = ff_index(c);
    if (atomicLoad(&parent[i]) == FF_NONE) { return; }
    let b = ff_shape(i).y;
    let sx = ff_sector(c.x, d.x);
    let l = ff_step(c, vec2i(-1, 0));
    if (!ff_white(l) || ff_sector(ff_pixel(u32(l)).x, d.x) != sx) { atomicOr(&box[b], 1u << sx); }
    let sy = ff_sector(c.y, d.y);
    let u = ff_step(c, vec2i(0, -1));
    if (!ff_white(u) || ff_sector(ff_pixel(u32(u)).y, d.y) != sy) { atomicOr(&box[b + 1u], 1u << sy); }
}
