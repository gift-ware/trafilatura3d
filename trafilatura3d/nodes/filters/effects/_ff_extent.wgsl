// Flood fill: the extents of each shape in its frame (ff_seam), taken where a run ends (a black or outside neighbour,
// or the frame's edge); `out` = 1 on each shape's first pixel (its root), else 0.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let i = ff_index(c);
    if (atomicLoad(&parent[i]) == FF_NONE) { store_out(c, fx_gray(0.0)); return; }
    let s = ff_shape(i);
    let b = s.y;
    let f = (c - vec2i(ff_seam(atomicLoad(&box[b]), d.x, tile_h()), ff_seam(atomicLoad(&box[b + 1u]), d.y, tile_v()))
             + d) % d;
    if (f.x == 0 || !ff_white(ff_step(c, vec2i(-1, 0)))) { atomicMin(&box[b + 2u], u32(f.x)); }
    if (f.x == d.x - 1 || !ff_white(ff_step(c, vec2i(1, 0)))) { atomicMax(&box[b + 3u], u32(f.x)); }
    if (f.y == 0 || !ff_white(ff_step(c, vec2i(0, -1)))) { atomicMin(&box[b + 4u], u32(f.y)); }
    if (f.y == d.y - 1 || !ff_white(ff_step(c, vec2i(0, 1)))) { atomicMax(&box[b + 5u], u32(f.y)); }
    store_out(c, fx_gray(select(0.0, 1.0, s.x == i)));
}
