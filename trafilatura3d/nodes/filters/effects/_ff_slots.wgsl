// Flood fill: each root takes the next slot of the shapes' values (`count`, which ends as the number of shapes) and
// marks it in its own entry (FF_ROOT | slot).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let i = ff_index(c);
    if (atomicLoad(&parent[i]) != i) { return; }
    atomicStore(&parent[i], FF_ROOT | atomicAdd(&count[0], 1u));
}
