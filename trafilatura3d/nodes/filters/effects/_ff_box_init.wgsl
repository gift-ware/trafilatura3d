// Flood fill: each root empties its shape's values (no sectors, extents to be narrowed by atomic min / max).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let up = atomicLoad(&parent[ff_index(c)]);
    if (up == FF_NONE || (up & FF_ROOT) == 0u) { return; }
    let b = (up & ~FF_ROOT) * FF_BOX;
    atomicStore(&box[b], 0u);
    atomicStore(&box[b + 1u], 0u);
    atomicStore(&box[b + 2u], FF_NONE);
    atomicStore(&box[b + 3u], 0u);
    atomicStore(&box[b + 4u], FF_NONE);
    atomicStore(&box[b + 5u], 0u);
}
